import time

def duration(fcn):
    
    def wrapper(*args):
        t0 = time.time()
        ret = fcn(*args)
        dt = time.time() - t0
        dt = round(dt, 3)

        # storeTime = kwargs.get('storeTime')
        # if storeTime:  storeTime = dt
        return (ret, dt)
    return wrapper
