from python.aggregation import (
    Aggregation,
    ArbitraryAggregation,
    PeriodicAggregation,
    TWO_DAYS,
    FOUR_DAYS,
    WEEKDAYS,
    WEEKENDS,
    WEEKS,
    FORTNIGHTS,
    MONTHS,
    DAYS
)
from python.temporal_hierarchy import TemporalHierarchy

AVAILABLE_LEVELS: list[Aggregation] = [
    TWO_DAYS,
    FOUR_DAYS,
    WEEKDAYS,
    WEEKENDS,
    WEEKS,
    FORTNIGHTS,
    ArbitraryAggregation('8Days1', [1] * 8 + [0] * 20),
    ArbitraryAggregation('8Days2', [0] * 8 + [1] * 8 + [0] * 12),
    ArbitraryAggregation('8Days3', [0] * 16 + [1] * 8 + [0] * 4),
    ArbitraryAggregation('16Days', [1] * 16 + [0] * 12),
    ArbitraryAggregation('12Days', [0] * 16 + [1] * 12),
    PeriodicAggregation('11L', 1, 1, True), # Skip every other
    PeriodicAggregation('11R', 1, 1, False),
    PeriodicAggregation('43L', 4, 3, True), # 4 Days on, 3 skip
    PeriodicAggregation('43R', 4, 3, False),
    PeriodicAggregation('34L', 3, 4, True), # 3 Days on, 4 skip
    PeriodicAggregation('34R', 3, 4, False),
    PeriodicAggregation('61L', 6, 1, True), # 6 Days on, 1 skip
    ArbitraryAggregation('AllWeekends', [0,0,0,0,0,1,1] * 4),
    ArbitraryAggregation('AllWeekdays', [1,1,1,1,1,0,0] * 4),
    ArbitraryAggregation('AllExtendedWeekends', [0,0,0,0,1,1,1] * 4),
    ArbitraryAggregation('AllExtendedWeekdays', [1,1,1,1,1,1,0] * 4),
    ArbitraryAggregation('10', [1, 0] * 14), # Zig zag
    ArbitraryAggregation('01', [0, 1] * 14)
]

ALL_POSSIBLE = TemporalHierarchy([MONTHS] + AVAILABLE_LEVELS + [DAYS],'AllPossible')
