from multiprocessing import Pool


def map_parallelize(func, it):
    pool = Pool(14)
    ret = pool.map(func, it)
    pool.close()
    pool.join()
    return ret
