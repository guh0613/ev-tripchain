import numpy as np

from ev_tripchain.mobility.soc import SOCEvolutionParams, simulate_soc_and_charging_profile
from ev_tripchain.mobility.trip_chain import Stop, TripChain


def test_soc_decreases_with_mileage_and_charges_when_triggered() -> None:
    # home(0-60) -> work(90-300) -> home(330-1440)
    stops = [
        Stop(zone=0, arrival_minute=0, departure_minute=60, purpose="home"),
        Stop(zone=1, arrival_minute=90, departure_minute=300, purpose="work"),
        Stop(zone=0, arrival_minute=330, departure_minute=1440, purpose="home"),
    ]
    tc = TripChain(stops=stops, leg_distance_km=[50.0, 50.0])

    rng = np.random.default_rng(1)
    params = SOCEvolutionParams(
        battery_capacity_kwh=50.0,
        consumption_kwh_per_km=0.2,  # 10kWh per 50km
        initial_soc_mean=0.25,
        initial_soc_std=0.0,
        charge_power_kw=10.0,
        charge_efficiency=1.0,
        charge_trigger_soc=0.3,
        charge_purposes=("home", "work"),
    )

    soc, p_kw = simulate_soc_and_charging_profile(
        tc, params, step_minutes=30, n_steps=48, rng=rng, initial_soc=0.25
    )

    assert soc.shape == (49,)
    assert p_kw.shape == (48,)
    assert (soc >= 0.0).all() and (soc <= 1.0).all()
    # Should charge at home/work because initial SOC <= trigger
    assert p_kw.max() > 0.0
    # After the first long trip (50km), SOC should not increase due to travel
    # but later it should recover due to charging.
    assert soc.min() <= 0.25
    assert soc[-1] >= soc[0]

