FREQ_MIN_DEFAULT = 20.0
FREQ_MAX_DEFAULT = 20000.0
FREQ_TICKS = [20,30,40,50,60,80,100,200,300,400,600,800,1000,2000,3000,4000,5000,6000,8000,10000,20000]

def fmt_freq_tick(x):
    if x >= 1000:
        v = int(round(x/1000.0))
        return f"{v}k" if (x % 1000 == 0) else f"{x/1000.0:.1f}k"
    return f"{int(x)}"
