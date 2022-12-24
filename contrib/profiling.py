from modmesh.onedim import euler1d
from _modmesh import stop_watch


shock_tube = euler1d.ShockTube()
shock_tube.build_constant(
    gamma=1.4,
    pressure1=1.0,
    density1=1.0,
    pressure5=0.1,
    density5=0.125,
)

shock_tube.build_numerical(xmin=-1, xmax=1, ncoord=21, time_increment=0.05)

# reset lap
stop_watch.lap()
shock_tube.calc_velocity2(0, 10)
print(f"calc_velocity2: \t{stop_watch.lap()}")

# reset lap
stop_watch.lap()
shock_tube.build_field(t=10)
print(f"build_field t=10:\t{stop_watch.lap()}")

# reset lap
stop_watch.lap()
shock_tube.build_field(t=100)
print(f"build_field t=100:\t{stop_watch.lap()}")
