from projection_utility import ProjectionInputMapper
import numpy
from scipy.stats import norm

class MiniColumnInputMapper(ProjectionInputMapper):
    '''
    An implementation of ProjectionInputMapper that maps to minicolumns
    '''    
    sigma = 1.0
    exclusion = 100.0    
    conduction_velocity = 300 #micron/ms, TODO: read from recipe?
    minicolumns = []        
    minicolumn_indices = []
    #TODO: number of gids per minicolumn! ATM fixed at 1
        
    def __init__(self,specs,cfg,absoluteVolume):
        ProjectionInputMapper.__init__(self,specs,cfg)        
        self.sigma = float(specs.get("sigma"))
        self.exclusion = float(specs.get("exclusion"))
        self.minicolumns = cfg.get_mvd_minicolumns()
        self.minicolumn_indices = numpy.nonzero(
                                                (self.minicolumns[:,0] > absoluteVolume[0]-self.exclusion) &
                                                (self.minicolumns[:,0] < absoluteVolume[1]+self.exclusion) &
                                                (self.minicolumns[:,1] > absoluteVolume[4]-self.exclusion) &
                                                (self.minicolumns[:,1] < absoluteVolume[5]+self.exclusion)
                                                )
        self.minicolumns = self.minicolumns[self.minicolumn_indices[0]]
        self.used_gid_offset = len(self.cfg.get_mvd_minicolumns())
        
    def resolve_exclusion(self,positions):
        return [numpy.nonzero([(x[0]<self.exclusion)&(x[1]<self.exclusion) for x in numpy.abs(self.minicolumns - loc)])
                for loc in positions]                    
        
    def resolve(self,positions):
        resIdx = self.resolve_exclusion(positions)        
        return [resIdx[i][0][self.pick_single(numpy.sqrt(numpy.sum((positions[i]-self.minicolumns[resIdx[i]])**2,axis=1)))]
                for i in range(len(positions))]
    
    def delay(self,y_positions):
        return y_positions/self.conduction_velocity
        
    def pick_single(self,distances):
        if len(distances)==0:
            return []
        distances = norm.pdf(distances,0,self.sigma)
        found = numpy.nonzero(numpy.random.random() <= (numpy.cumsum(distances)/numpy.sum(distances)))
        return found[0][0] 
    
    def get_mapping(self,syn_loc,seg_spec,syn_type_names):
        gid_offset, gid_max = self.gid_range()
        minicol_resolved = self.resolve(syn_loc[:,[0,2]])
        #import pdb
        #pdb.set_trace()
        for i in range(len(minicol_resolved)):
            if(type(minicol_resolved[i]) in [int, numpy.int64]):
                minicol_resolved[i] = self.minicolumn_indices[0][minicol_resolved[i]] + gid_offset
            else:
                minicol_resolved[i] = numpy.NaN
        if len(minicol_resolved)>0:
            mapped_gids = range(gid_offset,numpy.nanmax(minicol_resolved))
            self.mapping_counts = (numpy.histogram(minicol_resolved,numpy.hstack((mapped_gids,mapped_gids[-1]+1)))[0],mapped_gids)            
            self.used_gid_offset = numpy.max((mapped_gids[-1]-gid_offset+1,self.used_gid_offset))
        
        return numpy.vstack((minicol_resolved,self.delay(syn_loc[:,1]))).transpose()
                             
    
    
    
class RandomInputMapper(ProjectionInputMapper):
    '''
    An implementation of ProjectionInputMapper that maps randomly
    '''    
    extra_gid_offset = 1337    
    syns_per_gid = []
    num_assigned_gids = []    
        
    def __init__(self,specs,cfg):
        ProjectionInputMapper.__init__(self,specs,cfg)
        if 'num_assigned_gids' in specs.keys():
            self.num_assigned_gids = int(specs.get("num_assigned_gids"))
        else:
            self.syns_per_gid = int(specs.get("syns_per_gid"))        
        self.mapping_counts = []
        self._targets = []
    
    def get_mapping(self,syn_loc,seg_spec,syn_type_names):
        if syn_loc.shape[0] == 0:
            return numpy.vstack(([],[])).transpose()        
        IGNORE, gid_offset = self.gid_range()
        if len(self.num_assigned_gids)==0:
            c_p_g = self.syns_per_gid
            n_a_g = int(numpy.ceil(syn_loc.shape[0]/float(self.syns_per_gid)))
        else:
            n_a_g = self.num_assigned_gids
            c_p_g = int(numpy.ceil(syn_loc.shape[0]/float(self.num_assigned_gids)))
        resolved = numpy.hstack([numpy.ones(c_p_g)*i for i in range(n_a_g)]) + gid_offset
        numpy.random.shuffle(resolved)
        resolved = resolved[:syn_loc.shape[0]]
        mapped_gids = range(gid_offset,numpy.nanmax(resolved))
        self.mapping_counts = (numpy.histogram(resolved,numpy.hstack((mapped_gids,mapped_gids[-1]+1)))[0],mapped_gids)
        self._targets.append(("_Source" + str(len(self._targets)), (gid_offset,mapped_gids[-1]+1)))
        self.used_gid_offset += n_a_g
        
        return numpy.vstack((resolved,numpy.ones_like(resolved).astype(float))).transpose()
    def targets(self):
        return self._targets
    
