'''
Created on Jun 14, 2012

@author: reimann
'''

import os
import numpy
from bluepy import interact
import bbp
from scipy.stats import norm
import h5py
from lxml import etree

class SynapseClass(object):
    _u = (0.0,1.0)#NOTE: ALL PARAMETERS IN k, Theta!!!    
    _d = (0.0,1.0)
    _f = (0.0,1.0)        
    _g_max = (0.0,1.0)
    _dtc = (0.0,1.0)
    syn_type = 1337
    label = 'ultra_violent_synapse'        
    _ase = (0.0,1.0)
    def get_parameter_line(self,lgth):
        nrm = numpy.random.gamma
        shape = (lgth,1)
        return numpy.hstack((nrm(self._g_max[0],self._g_max[1],shape),
			     nrm(self._u[0],self._u[1],shape),
			     nrm(self._d[0],self._d[1],shape),
			     nrm(self._f[0],self._f[1],shape),
			     nrm(self._dtc[0],self._dtc[1],shape),
			     self.syn_type * numpy.ones(shape),
			     nrm(self._ase[0],self._ase[1],shape)))
                            
            
class SynTypeMap(object):
    _map = {}
        
    def __init__(self):
        self._map = {}
        
    def merge(self,other_map):
        for k,v in other_map._map.items():
            if self._map.has_key(k):
                self._map[k] = self._map[k] + v
            else:
                self._map[k] = v
                
    def put(self,k,v):
        if(type(v) != list or type(k) != str):
            print("INvalid!")#TODO: should be Exception
        else:
            self._map[k] = v
            
    def size(self):
        return numpy.sum([len(x) for x in self._map.values()])

def get_gamma_parameters(mn,sd):
    return ((mn/sd)**2, (sd**2)/mn) #k, theta or shape, scale

class SynapseClassifier(object):
    syn_classes = {}    
    
    def __init__(self,specs):
        for syn in specs:
            if syn.tag == 'SynapseType':
                new_type = SynapseClass()
                new_type.label = syn.get('label')
                new_type.syn_type = int(syn.get('id'))
                for p in syn:
                    if p.tag == 'parameter':
                        if p.get('id') == "gsyn":
                            new_type._g_max = get_gamma_parameters(float(p.get('mean')), float(p.get('std')))
                        elif p.get('id') == "Use":
                            new_type._u = get_gamma_parameters(float(p.get('mean')), float(p.get('std')))
                        elif p.get('id') == "D":
                            new_type._d = get_gamma_parameters(float(p.get('mean')), float(p.get('std')))
                        elif p.get('id') == "F":
                            new_type._f = get_gamma_parameters(float(p.get('mean')), float(p.get('std')))
                        elif p.get('id') == "Ase":
                            new_type._ase = get_gamma_parameters(float(p.get('mean')), float(p.get('std')))
                        elif p.get('id') == "dtc":
                            new_type._dtc = get_gamma_parameters(float(p.get('mean')), float(p.get('std')))
                self.syn_classes[new_type.label] = new_type
    
    def get_synapse_parameters(self,syn_type_name_mapper):
        mp = syn_type_name_mapper._map
        total_lgth = numpy.sum([len(x) for x in mp.values()])
        ret_val = numpy.zeros((total_lgth,7))
        for s,t in mp.iteritems():
            ret_val[t] = self.syn_classes[s].get_parameter_line(len(t))
        
        return ret_val                            
                                            

class ProjectionInputMapper(object):
    cfg = []
    mapping_counts = ([],[])
    max_circuit_gid = 0
    extra_gid_offset = 1000
    used_gid_offset = 0
    '''
    Abstract class to handle the mapping of projection synapses to virtual presynaptic cells
    '''
    def __init__(self,specs,cfg):
        self.cfg = cfg
        self.max_circuit_gid = len(self.cfg.get_target('Mosaic'))
        self.extra_gid_offset = 1000
        self.used_gid_offset = 0
    def gid_range(self):
        """ returns gid interval into which this mapping can map [min, max+1] """
        return (self.max_circuit_gid + self.extra_gid_offset, 
                self.max_circuit_gid + self.extra_gid_offset + self.used_gid_offset)
    def targets(self):
        """returns name and gids of projecting cell targets. Default: name is _Source and the whole gid range"""
        return [("_Source",self.gid_range())]
    def get_mapping(self,syn_loc,seg_spec,syn_type_names):
        '''
        This is the abstract function that has to be implemented. Returns a list of gids.
        Input: syn_loc: array of shape (N,3) with the location of N synapses [x,y,z]
               seg_spec: array of shape (N,3), for each synapse [presynGid, preSecId, preSegId]
               syn_type_names: list of length N, for each synapse the identifier of the synapse type
        '''
        raise NotImplementedError     

               

class NeuronalProjection(object):
    '''
    Abstract class for general NeuronalProjections
    '''
    
    class defaultParameters(object):
        post_segment_distance = 0.5
        pre_section_id = 0
        pre_segment_id = 0
        pre_segment_distance = 0.5
        pre_mType = 1337
        pre_bo = 1
        post_bo = 1        
    
    label = "Undefined projection"
    outfiles = []
    inputs = {}
    
    _data = []
    cfg = [];
    synapse_specs = []
    default = []
    
    def __init__(self,lbl,cfg,synapse_specs):
        self.label = lbl
        self.cfg = cfg
        self.synapse_specs = synapse_specs
        self.default = NeuronalProjection.defaultParameters()
        
    def cache_projection_synapses(self):
        (seg_spec,syn_type_name_mapper,pre_info) = self.get_projection_synapses()
        print("Equipping synapses with parameters...")
        syn_parameters = self.synapse_specs.get_synapse_parameters(syn_type_name_mapper)
        print("Assembling final data structura...")
        self._data = (seg_spec,syn_parameters,pre_info)
        self._data = self.get_final_h5data_structure(seg_spec,syn_parameters,pre_info)
        
    def write_h5_file(self,path,name,num_files):
        if(len(self._data)==0):
            print("Generating projection synapses...")
            self.cache_projection_synapses()   
        
        print("Done. Starting to write h5 file(s)")     
        (uGids,data_list) = self._data        
        data_list.reverse()
        
        #if num_files > len(uGids):
        #    num_files = len(uGids)
        splits = numpy.round(numpy.linspace(0,len(uGids),num_files+1))
        split_gids = numpy.split(uGids, splits[1:-1])        
        
        for i in range(num_files):
            filename = path + name + '.h5.' + str(int(i))
            if os.path.exists(filename): 
                raise IOError, "File %s exists." % filename
        
            h5file = h5py.File(filename,'w')
            t_gids = split_gids[i]
            for g in t_gids:
                dat = data_list.pop() 
                # skip if zero synapses
                if len(dat)==0: 
                    continue
                # sort data set
                idx = numpy.argsort(dat[:,0], kind='mergesort')
                sorted_dat = dat[idx,:]
                d_set = h5file.create_dataset('a' + str(int(g)), sorted_dat.shape, ">f4", sorted_dat)
                
            h5file.close()
                                
    def get_final_h5data_structure(self,seg_spec,syn_parameters,pre_info):                
        
        def inner_assembler(i):
            return numpy.hstack( (pre_info[i,:], seg_spec[i,1:3], 
                                  self.default.post_segment_distance, self.default.pre_section_id, self.default.pre_segment_id, #TODO: pre_section_id, pre_segment_id from pre_info???? 
                                  self.default.pre_segment_distance, syn_parameters[i,0:6], self.default.pre_mType, self.default.post_bo, #TODO: pre_mType from pre_info
                                  self.default.pre_bo, syn_parameters[i,6], seg_spec[i,3]) ) #TODO: pre_bo from pre_info
        
        uGids = numpy.unique(seg_spec[:,0])
        data_list = []
    #import pdb
    #pdb.set_trace()
        for g in uGids:
            idx = numpy.nonzero(seg_spec[:,0]==g)
            sort_idx = numpy.argsort(pre_info[idx[0],0])
            data_list.append(numpy.array([inner_assembler(i) for i in idx[0][sort_idx] if not numpy.isnan(pre_info[i,0])]))

        return (uGids,data_list)        
        
        
    def get_projection_synapses(self): #How do abstract methods work in python ?
        pass 
    
class VolumeProjection(NeuronalProjection):
    '''
    Implementation for volumetric projections
    '''
    volume_specs = []    
    mapping_specs = []
    idx_obj = []
        
    def __init__(self,proj_xml,cfg, syn_specs):
        '''
        Input: proj_xml, the xml specification of the projection of type volumetric projection, i.e. proj_xml.tag == 'Projection'
        and proj_xml.get('type') == 'volume projection'
               cfg, a bluepy.parsers.blueconfig.BlueConfig object
        
        '''
        self.idx_obj = cfg.segment_spatial_index()                        
        for i in range(0,len(proj_xml)):
            if(proj_xml[i].tag == "Volume"):
                self.volume_specs.append(VolumeSpecifiedSegmentTarget(proj_xml[i],cfg,self.idx_obj))
                
        absoluteVolume = numpy.vstack([x.volume_bounds for x in self.volume_specs])
        absoluteVolume = [numpy.min(absoluteVolume[:,0]),
                          numpy.max(absoluteVolume[:,1]),
                          numpy.min(absoluteVolume[:,2]),
                          numpy.max(absoluteVolume[:,3]),
                          numpy.min(absoluteVolume[:,4]),
                          numpy.max(absoluteVolume[:,5])]
        #TODO: the following might go into NeuronalProjection instead..?        
        for i in range(0,len(proj_xml)):
            if(proj_xml[i].tag == "InputMapping"):
                if(proj_xml[i].get('type') == 'MapToMiniColumns'):
                    from InputMappers import MiniColumnInputMapper
                    self.mapping_specs = MiniColumnInputMapper(proj_xml[i],cfg,absoluteVolume)
                if(proj_xml[i].get('type') == 'Random'):
                    from InputMappers import RandomInputMapper
                    self.mapping_specs = RandomInputMapper(proj_xml[i],cfg)
                else:
                    print("unknown mapping type")#TODO: should be exception 
                
        NeuronalProjection.__init__(self,proj_xml.get('id'),cfg,syn_specs)                    
    
    def get_projection_synapses(self):
        seg_spec = numpy.zeros((0,4))
        #syn_type_names = []
        syn_type_name_mapper = SynTypeMap()
        pre_info = numpy.zeros((0,2))        
        
        index_offset = 0
        
        for i in range(0,len(self.volume_specs)):
            (syn_loc,append_me,type_ids) = numpy.split(self.volume_specs[i].find_segments(),[3,7],axis=1)
            for tp in numpy.unique(type_ids):
                found_idx = numpy.nonzero(type_ids==tp)
                syn_type_name_mapper.merge(self.volume_specs[i].valid_touches[int(tp)].get_syn_type_map(found_idx[0]+index_offset))
            index_offset = index_offset + syn_type_name_mapper.size()
            #type_ids = [int(x) for x in type_ids]
            #type_strings = [self.volume_specs[i].valid_touches[x].get_syn_type_string() for x in type_ids]
            
            seg_spec = numpy.vstack((seg_spec,append_me))            
            #syn_type_names += type_strings                                               
            pre_info = numpy.vstack((pre_info,self.mapping_specs.get_mapping(syn_loc,seg_spec,syn_type_name_mapper)))
        
        return (seg_spec,syn_type_name_mapper,pre_info)         
        
    
class VolumeSpecifiedSegmentTarget(object):
    '''
    Helper class for VolumeProjection that resolves a single volume (a projection can contain >1 volume)
    '''
    class SegmentTargetSpec(object):
        '''
        This class checks the validity of segments found, i.e. section type and mtype. Also assigns synapse types. 
        '''
        sectionIds = []
        gids = [] #Note: It actually compares against gids. this is faster than first looking up the mtype for each segment found!
        synTypes = []
        synType_probs = []
        synType_cumulative_probs = []
        preference = 0.0
        
        def __init__(self):
            self.sectionIds = []
            self.gids = [] #Note: It actually compares against gids. this is faster than first looking up the mtype for each segment found!
            self.synTypes = []
            self.synType_probs = []
            self.synType_cumulative_probs = []
            self.preference = 0.0
        
        def check(self,gids_toCheck,secTypes_toCheck):
            '''
            Checks the validity of a segment
            '''
            valid_ids = numpy.nonzero(numpy.vectorize(lambda x: x in self.sectionIds)(secTypes_toCheck) &
            numpy.vectorize(lambda x: x in self.gids)(gids_toCheck))
            return valid_ids
        
        def get_syn_type_string(self):
            '''
            randomly assigns a synapse type based on the probabilities in the recipe
            '''            
            idx = numpy.nonzero((self.synType_cumulative_probs - numpy.random.random()) >= 0)
            return self.synTypes[idx[0][0]]
        
        def get_syn_type_map(self,numbers):
            my_random_numbers = numpy.random.random(len(numbers))            
            c_prob = self.synType_cumulative_probs.tolist()
            c_prob.reverse()
            c_prob.append(-1.0)
            c_prob.reverse()
            map_to_return = SynTypeMap()
            for i in range(len(self.synTypes)):
                found = numpy.nonzero((c_prob[i]< my_random_numbers) & (c_prob[i+1] >= my_random_numbers))
                map_to_return.put(self.synTypes[i], numbers[found[0]].tolist())
            
            return map_to_return
        
        def add_syn_type(self, type_string, type_prob):
            self.synTypes.append(type_string)
            self.synType_probs.append(type_prob)
            self.synType_cumulative_probs = numpy.cumsum(self.synType_probs)/numpy.sum(self.synType_probs)
        
            
    volume_bounds = [] #x_from,x_to,y_from,y_to,z_from,z_to
    density_bins = []
    density_values = []
    valid_touches = []
    
    epsilon = 0.0001
    cfg = []
    idx_obj = []
        
    synType_map = {}
    
    def __init__(self,proj_xml,cfg,idx):
        from numpy import nan as NaN
        self.volume_bounds = [NaN,NaN,NaN,NaN,NaN,NaN]
        self.valid_touches = []
        self.idx_obj = idx
        self.cfg = cfg        
        func_map = {}
        func_map['Boundaries'] = self.treat_boundary_spec
        func_map['DensityProfile'] = self.treat_density_spec
        func_map['Targets'] = self.treat_target_spec
        for i in numpy.arange(0,len(proj_xml)):
            spec_element = proj_xml[i]
            if func_map.has_key(spec_element.tag):
                func_map[spec_element.tag](spec_element)
            
    def treat_boundary_spec(self,specs):
        layers = self.cfg.recipe.xpath("/blueColumn/column/layer")
        layerIds = numpy.array([int(x.get("id")) for x in layers])        
        layerIndices = numpy.argsort(layerIds)        
        layerIds = layerIds[layerIndices[::-1]]
        thickness = numpy.array([float(x.get("thickness")) for x in layers])
        layer_bounds = numpy.hstack((0,thickness[layerIndices[::-1]].cumsum()))
        for i in numpy.arange(0,len(specs)):
            if(specs[i].get("type") == "layer"):
                l = numpy.nonzero([x==int(specs[i].get("id")) for x in layerIds])
                bounds = layer_bounds[numpy.hstack((l[0],l[0]+1))]
                rel = float(specs[i].get("rel"))
                if(specs[i].get("which") == "y_max"):                    
                    self.volume_bounds[3] = numpy.nanmin(((1-rel) * bounds[0] + rel * bounds[1],self.volume_bounds[3]))                                    
                elif(specs[i].get("which") == "y_min"):                         
                    self.volume_bounds[2] = numpy.nanmax(((1-rel) * bounds[0] + rel * bounds[1],self.volume_bounds[2]))                
            elif(specs[i].get("type") == "mosaic"):
                circ_geo = self.cfg.get_mosaic_geometry()
                hexes = [int(x) for x in specs[i].get("hex").split(',')]
                extents = numpy.vstack([circ_geo.mosaic_dict[circ_geo.coords_by_id[hex_id]].extent() for hex_id in hexes])
                #extents[:,[0,2]] -= circ_geo.ll[0]
                #extents[:,[1,3]] -= circ_geo.ll[1]
                if specs[i].get("which") == "x_min":
                    self.volume_bounds[0] = numpy.nanmax((numpy.min(extents[:,0]),self.volume_bounds[0]))
                elif specs[i].get("which") == "x_max":
                    self.volume_bounds[1] = numpy.nanmin((numpy.max(extents[:,2]),self.volume_bounds[1]))
                elif specs[i].get("which") == "z_min":
                    self.volume_bounds[4] = numpy.nanmax((numpy.min(extents[:,1]),self.volume_bounds[4]))
                elif specs[i].get("which") == "z_max":
                    self.volume_bounds[5] = numpy.nanmin((numpy.max(extents[:,3]),self.volume_bounds[5]))
                #self.volume_bounds[0] = 500#346 
                #self.volume_bounds[1] = 525#809
                #self.volume_bounds[4] = 300#600
                #self.volume_bounds[5] = 325#1000
                    
    def treat_density_spec(self,specs):
        if specs.get("type") == "y_rel":
            bin_heights = numpy.array([(x.get("height")) for x in specs])
            bin_heights = numpy.array([float(x) for x in bin_heights[bin_heights.nonzero()]])
            self.density_bins = numpy.zeros((len(bin_heights),2))
            bin_heights = numpy.hstack((0,(bin_heights[0:-1] + bin_heights[1:])/2,1))
            self.density_bins[:,0] = bin_heights[0:-1]
            self.density_bins[:,1] = bin_heights[1:]
            self.density_values = numpy.array([(x.get("density")) for x in specs])
            self.density_values = numpy.array([float(x) for x in self.density_values[self.density_values.nonzero()]])
            
            
    def treat_target_spec(self,specs):
        sec_type_map = bbp.Section_Type.names
                
        def assemble_target_spec(tgt):
            seg_tgt = VolumeSpecifiedSegmentTarget.SegmentTargetSpec()
            seg_tgt.sectionIds = []            
            seg_tgt.gids = set([])
            seg_tgt.preference = float(tgt.get("preference"))
            for x in tgt:
                if x.tag == 'synType':
                    seg_tgt.add_syn_type(x.get("id"),float(x.get("fraction")))
                elif x.tag == 'cell_type':
                    valid_gids = self.cfg.get_target("Mosaic")
                    if(x.get('eType')):
                        raise NotImplementedError
                    elif(x.get('mType')):
                        valid_gids = valid_gids.intersection(self.cfg.get_target(x.get('mType')))
                    elif(x.get('cell_target')):
                        valid_gids = valid_gids.intersection(self.cfg.get_target(x.get('cell_target')))
                    seg_tgt.gids = seg_tgt.gids.union(valid_gids)
                elif x.tag == 'region':
                    seg_tgt.sectionIds.append(sec_type_map[x.get("secType")])
            #seg_tgt.synTypes = numpy.array([x.get("id") for x in tgt])
            #seg_tgt.synTypeProbs = numpy.array([x.get("fraction") for x in tgt])
            #seg_tgt.synTypeProbs = numpy.array([float(x) for x in seg_tgt.synTypeProbs[seg_tgt.synTypes.nonzero()]])
            #seg_tgt.synTypes = seg_tgt.synTypes[seg_tgt.synTypes.nonzero()]
            return seg_tgt
                
        self.valid_touches = [assemble_target_spec(x) for x in specs.iterchildren() if x.tag=='target']
        
    
    def find_segments(self):
        seg_spec = numpy.zeros((0,8))        
        
        total_vol = (self.volume_bounds[1]-self.volume_bounds[0])*(self.volume_bounds[3]-self.volume_bounds[2])*(self.volume_bounds[5]-self.volume_bounds[4])        
        for i in numpy.arange(0,len(self.density_bins)):
            local_y_from = (1-self.density_bins[i,0])*self.volume_bounds[2] + self.density_bins[i,0]*self.volume_bounds[3]
            local_y_to = (1-self.density_bins[i,1])*self.volume_bounds[2] + self.density_bins[i,1]*self.volume_bounds[3] - self.epsilon
            seg_candidates = self.idx_obj.q_window_oncenter((self.volume_bounds[0],local_y_from,self.volume_bounds[4]),(self.volume_bounds[1],local_y_to,self.volume_bounds[5]))
            new_segs = self.filter_segments(seg_candidates,numpy.round(total_vol*self.density_values[i]*(self.density_bins[i,1]-self.density_bins[i,0])))                         
            seg_spec = numpy.vstack((seg_spec,new_segs))
        
        return seg_spec
            
            
    def filter_segments(self,seg_candidates,number_to_pick):
        valid = numpy.zeros(seg_candidates.shape[0])
        passed_rule = -numpy.ones(seg_candidates.shape[0],dtype=int)        
        #mTypes = self.cfg.mvddb.get_neurons_attribute(seg_candidates[:,8],"mType_id")
        for i in numpy.arange(len(self.valid_touches)):            
            valid_ids = self.valid_touches[i].check(seg_candidates[:,8],seg_candidates[:,11])
            valid[valid_ids] = valid[valid_ids] + self.valid_touches[i].preference
            passed_rule[valid_ids] = i
                        
        valid_starts = seg_candidates[valid>0,0:3]
        valid_ends = seg_candidates[valid>0,3:6]
        candidate_scores = numpy.sqrt(numpy.sum((valid_starts-valid_ends)**2,axis=1)) * valid[valid>0]
        seg_candidates = seg_candidates[valid>0,:]
        passed_rule = passed_rule[valid>0] 
        picked_indices = self.pick_candidates(candidate_scores, number_to_pick)
        seg_center_coords = (seg_candidates[picked_indices,(0,1,2)] + seg_candidates[picked_indices,(3,4,5)])/2        
        seg_spec = numpy.hstack((seg_center_coords,seg_candidates[picked_indices,(8,9,10,11)],passed_rule[picked_indices]))                        
        
        return seg_spec
        
    def pick_candidates(self, candidate_scores, number_to_pick):
        #indices = self.candidate_scores.argsort(axis=0)
        candidate_scores = numpy.cumsum(candidate_scores)
        my_random_numbers = numpy.random.random((number_to_pick,1)) * candidate_scores[-1]
        my_random_numbers.sort(axis=0)
        j = 0
        picked = numpy.ones((number_to_pick,1))*numpy.NAN
        for i in numpy.arange(0,number_to_pick):
            while candidate_scores[j] < my_random_numbers[i]:
                j=j+1
            picked[i] = j
        
        return picked.tolist()
        
                        
                        
    
    

class ProjectionComposer(object):
    '''
    classdocs
    '''
    cfg_file = "/path/to/structural_cfg"
    xml_file = "/path/to/recipe"
    cfg = []
    proj_list = []
    #idx_obj = {}

    def __init__(self, cfg_file, xml_file):
        '''
        Constructor        
        '''
        self.cfg_file = cfg_file
        self.cfg = interact.load_circuit(cfg_file)
        #self.idx_obj = self.cfg.get_segment_spatial_index()
        self.xml_file = xml_file

    def write_proj_target(self, path, name, gid_range):
        from bluepy.parsers import target
        gid_min, gid_max = gid_range
        
        t = target.Target("proj_%s" % name, "Cell", target.to_gids(range(gid_min, gid_max)))

        with file(path, "a") as f:
            print >>f, t

        
        
        
    def write(self,path,num_files):
        xpo = etree.parse(self.xml_file)
        projections = xpo.xpath('/Projections')
        projections = projections[0]
        index_of_syn_spec = numpy.nonzero([x.tag=="Synapses" for x in projections])
        print("Building SynapseClassifier...")
        syn_spec = SynapseClassifier(projections[index_of_syn_spec[0][0]])
        print("...done")
                        
        max_used_gid_offset = 1        
        
        self.proj_list = []
        for proj in projections:
            if (proj.tag == "Projection") & (proj.get("type") == "volume projection"):
                print("Building a new projection...")
                new_proj = VolumeProjection(proj,self.cfg,syn_spec)
                gidOffset = proj.xpath("InputMapping/@gidOffset")
                if gidOffset!=[]:
                    print("Found ABSOLUTE gidOffset=%d" % gidOffset[0])
                    new_proj.mapping_specs.extra_gid_offset = gidOffset[0]
                else:
                    gidOffset = proj.xpath("InputMapping/@relOffset")
                    if gidOffset!=[]:
                        print("Found RELATIVE gidOffset=%d" % gidOffset[0])
                        new_proj.mapping_specs.extra_gid_offset = max_used_gid_offset + gidOffset[0]
                    else:
                        new_proj.mapping_specs.extra_gid_offset = max_used_gid_offset + 0
                print("..done. Writing...")
                self.proj_list.append(new_proj)
                new_proj.write_h5_file(path, 'proj_nrn', num_files)
                max_used_gid_offset = numpy.max((max_used_gid_offset,
                                                 new_proj.mapping_specs.extra_gid_offset + new_proj.mapping_specs.used_gid_offset))                
                print("done.")
                name = proj.get("id").strip()
                for tgt in new_proj.targets():
                    self.write_proj_target(os.path.join(path, "user.target"),
                                           name.replace(" ", "_") + tgt[0], tgt[1] )

            elif proj.tag == "Projection":
                raise RuntimeError
        
            
        
