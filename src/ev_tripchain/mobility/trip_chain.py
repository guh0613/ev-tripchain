from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stop:
    zone: int
    arrival_minute: int
    departure_minute: int
    purpose: str


@dataclass(frozen=True)
class TripChain:
    """
    Trip-chain container.

    `stops` defines the daily sequence of activities. Each trip leg is the movement
    from stop i -> stop i+1 with distance `leg_distance_km[i]`.
    """

    stops: list[Stop]
    leg_distance_km: list[float]

    def __post_init__(self) -> None:
        if len(self.stops) < 2:
            raise ValueError("TripChain requires at least 2 stops.")
        if len(self.leg_distance_km) != (len(self.stops) - 1):
            raise ValueError("leg_distance_km length must be len(stops) - 1.")
        for s in self.stops:
            if s.arrival_minute > s.departure_minute:
                raise ValueError("Stop arrival_minute must be <= departure_minute.")
        for i in range(len(self.stops) - 1):
            if self.stops[i].departure_minute > self.stops[i + 1].arrival_minute:
                raise ValueError("Stops must be time-ordered (depart <= next arrival).")

    @property
    def n_legs(self) -> int:
        return len(self.stops) - 1
