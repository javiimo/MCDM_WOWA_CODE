import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Iterable, Set
from datetime import date, timedelta
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from project_paths import ERROR_LOG_PATH, ensure_output_tree

####################################################
# Auxiliary functions
####################################################

# Set up the logger for errors and warnings
logger = logging.getLogger("error_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    ensure_output_tree()
    fh = logging.FileHandler(ERROR_LOG_PATH)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(funcName)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def compute_stats(data: List[float]) -> str:
    """Compute basic statistics (min, max, mean, stdev) for a list of numbers."""
    if not data:
        return "no data"
    try:
        avg = statistics.mean(data)
        deviation = statistics.stdev(data) if len(data) > 1 else 0
        return (
            f"min={min(data)}, max={max(data)}, mean={avg:.2f}, stdev={deviation:.2f}"
        )
    except Exception as e:
        return f"error: {e}"


def trim_list(lst: List[Any]) -> str:
    """
    Return a string representation of a list. If the list has more than 20 elements,
    only the first 20 are shown followed by an ellipsis.
    """
    if not isinstance(lst, list):
        return str(lst)
    if len(lst) > 20:
        return str(lst[:20]) + " ... (total: " + str(len(lst)) + " elements)"
    return str(lst)


######################################################
#               Managing Windows Functions
######################################################


# Seasonal blocks (t_start, t_end, D_min, D_max): winter, interseason, summer,
# interseason, winter. Defaults correspond to X12; see configure_season_blocks.
SEASON_BLOCKS = [
    (1, 40, 1, 59),
    (41, 57, 60, 120),
    (58, 104, 121, 243),
    (105, 114, 244, 304),
    (115, 131, 305, 365),
]
BLOCK_DAY_RANGES = [(1, 59), (60, 120), (121, 243), (244, 304), (305, 365)]


def configure_season_blocks(seasons: Dict[str, List[Any]]) -> None:
    """
    Derive the seasonal blocks of an instance from its "Seasons" field, assuming the
    ROADEF order winter, interseason, summer, interseason, winter.
    """
    labels = {int(t): name for name, periods in seasons.items() for t in periods}
    runs = []
    for t in sorted(labels):
        if runs and labels[t] == runs[-1][2] and t == runs[-1][1] + 1:
            runs[-1][1] = t
        else:
            runs.append([t, t, labels[t]])
    if len(runs) != len(BLOCK_DAY_RANGES):
        raise ValueError(f"Unexpected season structure: {runs}")
    SEASON_BLOCKS[:] = [
        (start, end, d_min, d_max)
        for (start, end, _), (d_min, d_max) in zip(runs, BLOCK_DAY_RANGES)
    ]


def timestep_to_day_interval(t: int) -> Tuple[int, int]:
    """
    Map a timestep to a closed day-of-year interval [D_start, D_end] (1..365).

    Season convention (X12 default, see SEASON_BLOCKS):
        1–40    : winter       (January–February)
        41–57   : interseason1 (March–April)
        58–104  : summer       (May–August)
        105–114 : interseason2 (September–October)
        115–131 : winter2      (November–December)

    The days of each seasonal block are partitioned into contiguous integer
    intervals, one per timestep in that block. Interval lengths may differ
    by at most 1 day within a block.
    """
    for t_start, t_end, D_min, D_max in SEASON_BLOCKS:
        if t_start <= t <= t_end:
            break
    else:
        raise ValueError(f"t={t} outside the planning horizon")

    # Number of timesteps in this seasonal block
    N = t_end - t_start + 1
    # Total number of calendar days in this block (inclusive)
    total_days = D_max - D_min + 1

    if N > total_days:
        # More timesteps than days (e.g. X13): contiguous fractional intervals,
        # mapped to the days they intersect.
        idx = t - t_start
        return D_min + idx * total_days // N, D_min + ((idx + 1) * total_days - 1) // N

    # Base length of each timestep interval (integer division)
    base_len = total_days // N
    # Number of intervals that receive one extra day (to cover all days)
    extra = total_days % N  # the first `extra` intervals will have length base_len + 1

    # Zero-based index of this timestep inside its block
    idx = t - t_start

    # Total number of days assigned to previous intervals within this block
    # - each previous interval contributes at least base_len
    # - and the first `extra` of them contribute one additional day
    days_before = idx * base_len + min(idx, extra)

    # Start of this interval in day-of-year coordinates
    start = D_min + days_before
    # Length of this interval
    length = base_len + (1 if idx < extra else 0)
    # End of the interval (inclusive)
    end = start + length - 1

    return start, end


def model_day_to_date_2025(model_day: int) -> date:
    """
    Convert a model day-of-year index (1..365, without leap day)
    to the corresponding calendar date in year 2025.

    Mapping:
        model 1..365 -> 2025-01-01 .. 2025-12-31
    """
    if not 1 <= model_day <= 365:
        raise ValueError("model_day must be between 1 and 365")

    return date(2025, 1, 1) + timedelta(days=model_day - 1)


def describe_model_day_2025(model_day: int) -> dict:
    """
    Return calendar information for a given model day-of-year (1..365) in 2025.
    """
    d = model_day_to_date_2025(model_day)
    return {
        "date": d,  # datetime.date(2025, m, d)
        "weekday": d.weekday(),  # 0 = Monday, ..., 6 = Sunday
        "is_weekend": d.weekday() >= 5,
        "month": d.month,  # 1..12
        "day": d.day,  # day of month
        "iso_calendar": d.isocalendar(),  # (ISO year, ISO week number, ISO weekday)
    }


def date_to_model_day_2025(d: date) -> int:
    """
    Projects a calendar date (any year) to the model day 1..365,
    using only month and day (year fixed to 2025).

    Note: some dates may not be representable in the model
    (e.g. 29/02, 01/03 according to your mapping); in that case throws ValueError.
    """
    d2025 = date(2025, d.month, d.day)
    for md in range(1, 366):
        if model_day_to_date_2025(md) == d2025:
            return md
    raise ValueError(f"{d2025} is not representable in the model (no image in 1..365).")


# ----------------------------------------------------------------------
# Sets of model days by category
# ----------------------------------------------------------------------

ALL_MODEL_DAYS = range(1, 366)

# Winter: months of January, February, November and December 2025
WINTER_MODEL_DAYS: Set[int] = {
    md for md in ALL_MODEL_DAYS if model_day_to_date_2025(md).month in (1, 2, 11, 12)
}

# Summer: months of May to August 2025
SUMMER_MODEL_DAYS: Set[int] = {
    md for md in ALL_MODEL_DAYS if model_day_to_date_2025(md).month in (5, 6, 7, 8)
}

# National holidays 2025 (Metropolitan France)
# Source: Service-Public.fr "Les jours fériés en 2025" / "Jours fériés et ponts dans le secteur privé".
NATIONAL_HOLIDAY_DATES_2025 = [
    date(2025, 1, 1),  # New Year's Day
    date(2025, 4, 21),  # Easter Monday
    date(2025, 5, 1),  # Labour Day
    date(2025, 5, 8),  # Victory in Europe Day
    date(2025, 5, 29),  # Ascension Day
    date(2025, 6, 9),  # Whit Monday
    date(2025, 7, 14),  # Bastille Day
    date(2025, 8, 15),  # Assumption of Mary
    date(2025, 11, 1),  # All Saints' Day
    date(2025, 11, 11),  # Armistice Day 1918
    date(2025, 12, 25),  # Christmas Day
]

NATIONAL_HOLIDAY_MODEL_DAYS: Set[int] = set()
for d in NATIONAL_HOLIDAY_DATES_2025:
    try:
        NATIONAL_HOLIDAY_MODEL_DAYS.add(date_to_model_day_2025(d))
    except ValueError:
        # If a date is not representable in the model, we ignore it.
        pass

# Christmas 2025:
# - Christmas holidays 2024-2025: from 21/12/2024 to 06/01/2025
#   -> in 2025 we take 01/01/2025–06/01/2025
# - Christmas holidays 2025-2026: from 20/12/2025 to 05/01/2026
#   -> in 2025 we take 20/12/2025–31/12/2025
# Sources: arrêté du 7 décembre 2022 + synthesized school calendars.
CHRISTMAS_DATES_2025: Set[date] = set()

# January 1–6, 2025
d = date(2025, 1, 1)
while d <= date(2025, 1, 6):
    CHRISTMAS_DATES_2025.add(d)
    d += timedelta(days=1)

# December 20–31, 2025
d = date(2025, 12, 20)
while d <= date(2025, 12, 31):
    CHRISTMAS_DATES_2025.add(d)
    d += timedelta(days=1)

CHRISTMAS_MODEL_DAYS: Set[int] = set()
for d in CHRISTMAS_DATES_2025:
    try:
        CHRISTMAS_MODEL_DAYS.add(date_to_model_day_2025(d))
    except ValueError:
        pass

# Ski high season 2025:
# Defined as:
#   - Christmas period (CHRISTMAS_DATES_2025), and
#   - winter holidays 2025, union of zones A/B/C:
#       - Zone B : 08/02/2025 – 24/02/2025
#       - Zone C : 15/02/2025 – 03/03/2025
#       - Zone A : 22/02/2025 – 10/03/2025
#   -> approximate union: 08/02/2025 – 10/03/2025
# Sources: Service-Public.fr "Vacances d'hiver 2025 : les dates zone par zone" +
#          summaries of the 2024-2025 school calendar.
SKI_HIGH_DATES_2025: Set[date] = set(CHRISTMAS_DATES_2025)

d = date(2025, 2, 8)
while d <= date(2025, 3, 10):
    SKI_HIGH_DATES_2025.add(d)
    d += timedelta(days=1)

SKI_HIGH_MODEL_DAYS: Set[int] = set()
for d in SKI_HIGH_DATES_2025:
    try:
        SKI_HIGH_MODEL_DAYS.add(date_to_model_day_2025(d))
    except ValueError:
        # For example, if 01/03/2025 has no image in the model.
        pass


# ----------------------------------------------------------------------
# Functions for timestep ↔ model day set intersection
# ----------------------------------------------------------------------


def timestep_intersects_model_days(t: int, model_days: Iterable[int]) -> bool:
    """
    Returns True if the interval [D_start, D_end] of timestep t
    contains at least one model day from 'model_days'.
    """
    D_start, D_end = timestep_to_day_interval(t)
    return any(D_start <= md <= D_end for md in model_days)


def timestep_has_pp1_pp2(t: int, pp1_pp2_dates: Iterable[date]) -> bool:
    """
    Returns True if any PP1/PP2 date from the last 12 months
    falls within the day interval associated with timestep t.

    The dates are projected to the model year 2025 using month and day.
    """
    pp_model_days: Set[int] = set()
    for d in pp1_pp2_dates:
        try:
            pp_model_days.add(date_to_model_day_2025(d))
        except ValueError:
            # If a date is not representable in the model, it is ignored.
            pass
    return timestep_intersects_model_days(t, pp_model_days)


def timestep_is_winter(t: int) -> bool:
    """True if timestep t overlaps with any day in January, February, November or December."""
    return timestep_intersects_model_days(t, WINTER_MODEL_DAYS)


def timestep_is_summer(t: int) -> bool:
    """True if timestep t overlaps with any day from May to August."""
    return timestep_intersects_model_days(t, SUMMER_MODEL_DAYS)


def timestep_is_christmas(t: int) -> bool:
    """True if timestep t overlaps with the Christmas period (school holidays)."""
    return timestep_intersects_model_days(t, CHRISTMAS_MODEL_DAYS)


def timestep_has_national_holiday(t: int) -> bool:
    """True if timestep t includes at least one 2025 national holiday."""
    return timestep_intersects_model_days(t, NATIONAL_HOLIDAY_MODEL_DAYS)


def timestep_is_ski_high_season(t: int) -> bool:
    """True if t falls in the ski high season (Christmas + winter holidays)."""
    return timestep_intersects_model_days(t, SKI_HIGH_MODEL_DAYS)


def timestep_has_weekend(t: int) -> bool:
    """
    True if the day interval represented by timestep t contains any Saturday or Sunday.

    This helper intentionally works from the closed model-day interval returned by
    `timestep_to_day_interval`, because a timestep represents a block of calendar
    days rather than a single date.
    """
    d_start, d_end = timestep_to_day_interval(t)
    base_date = date(2025, 1, 1)

    for model_day in range(d_start, d_end + 1):
        current_date = base_date + timedelta(days=model_day - 1)
        if current_date.weekday() >= 5:
            return True
    return False


######################################################
#           Data Classes
######################################################


@dataclass
class Resource:
    """
    Represents a resource available for maintenance.
    """

    name: str
    capacity_min: List[float]  # One value per period (length T)
    capacity_max: List[float]  # One value per period (length T)

    def __str__(self) -> str:
        stats_min = compute_stats(self.capacity_min)
        stats_max = compute_stats(self.capacity_max)
        return (
            f"Resource '{self.name}':\n"
            f"  Periods (T): {len(self.capacity_min)}\n"
            f"  Capacity min: {self.capacity_min} ({stats_min})\n"
            f"  Capacity max: {self.capacity_max} ({stats_max})"
        )


@dataclass
class Season:
    """
    Represents a season grouping certain time periods.
    """

    name: str
    periods: List[int]  # Time periods belonging to this season

    def __str__(self) -> str:
        if self.periods:
            return (
                f"Season '{self.name}': {len(self.periods)} period(s), "
                f"range: {min(self.periods)}-{max(self.periods)}, periods: {self.periods}"
            )
        else:
            return f"Season '{self.name}': no periods assigned."


@dataclass
class Exclusion:
    """
    Represents a mutual exclusion constraint: the given interventions
    must not overlap during the specified season.
    """

    interventions: List[str]
    season: str

    def __str__(self) -> str:
        return (
            f"Exclusion Constraint: Interventions {self.interventions} "
            f"cannot overlap during season '{self.season}'."
        )


@dataclass
class InterventionScheduleOption:
    """
    Represents one scheduling option for an intervention when started at a specific time.

    Attributes:
        start_time: The potential start time (e.g., day or period number).
        duration: The length of the intervention when started at this time.
        workloads: A mapping from resource name to a list of workload values
                   over the duration of the intervention. For example, if resource "c1"
                   is used only in the first period, it might be: {"c1": [14, 0, 0]}.
        risks: A list (over the intervention's timesteps) where each element is a
               list of risk values—one per scenario—for that period.
               For example: [[4,8,2], [0,0], [0,0,0]].
        mean_risk: A list of mean risk values calculated for each timestep for THIS start time across all scenarios.
                   Only using non-zero risks.
        worst_risk: The highest risk value across all timesteps and scenarios.
        total_workload: The sum of all workload values across all resources and timesteps and multiply by the duration
    """

    start_time: int
    duration: int
    workloads: Dict[str, List[float]]
    risks: List[List[float]]
    mean_risk: List[float] = field(default_factory=list)
    worst_risk: float = field(
        default=0.0
    )  # Highest risk across timesteps and scenarios
    total_workload: float = field(default=0.0)  # New attribute to store total workload

    def __post_init__(self):
        if self.risks:
            self.mean_risk = []
            for timestep_risks in self.risks:
                # Only consider non-zero risk values.
                non_zero_risks = [value for value in timestep_risks if value > 0]
                total_risk = sum(non_zero_risks)
                total_scenarios = len(non_zero_risks)
                mean = total_risk / total_scenarios if total_scenarios > 0 else 0.0
                self.mean_risk.append(mean)
            # Compute worst_risk as the maximum risk value across all timesteps and scenarios.
            all_risks = [value for timestep in self.risks for value in timestep]
            self.worst_risk = max(all_risks) if all_risks else 0.0
        else:
            self.mean_risk = []
            self.worst_risk = 0.0

        # Compute total_workload by summing all workload values in all lists.
        self.total_workload = (
            sum(sum(workload) for workload in self.workloads.values()) * self.duration
            if self.workloads
            else 0.0
        )

    def __str__(self) -> str:
        # Format workloads using compute_stats (assumed to be defined elsewhere).
        if self.workloads:
            workloads_str = "\n    ".join(
                f"{res}: {vals} ({compute_stats(vals)})"
                for res, vals in self.workloads.items()
            )
        else:
            workloads_str = "None"
        # Format the risk profiles per timestep.
        if self.risks:
            risks_str = "\n    ".join(
                f"Relative period {i + 1}: {r} ({compute_stats(r)})"
                for i, r in enumerate(self.risks)
            )
        else:
            risks_str = "None"
        # Properly format the mean_risk list.
        mean_risk_str = (
            "[" + ", ".join(f"{mr:.2f}" for mr in self.mean_risk) + "]"
            if self.mean_risk
            else "[]"
        )
        return (
            f"Schedule Option (start_time={self.start_time}, duration={self.duration}, mean_risk={mean_risk_str}, total_workload={self.total_workload}):\n"
            f"  Workloads:\n    {workloads_str}\n"
            f"  Risks:\n    {risks_str}"
        )


@dataclass
class Intervention:
    """
    Represents an intervention (maintenance task) to be scheduled.

    Attributes:
        name: Unique identifier for the intervention.
        latest_start_timestep: The last period at which the intervention may start so that it finishes on time.
        options: A list of possible scheduling options for the intervention.
        mean_risk_profile: A list of mean risk values calculated for each timestep across all options (i.e. across all start times).
    """

    name: str
    latest_start_timestep: int
    options: List[InterventionScheduleOption] = field(default_factory=list)
    mean_risk_profile: List[float] = field(init=False, default_factory=list)
    mean_workload_size: float = field(init=False, default=0.0)

    def add_option(self, option: InterventionScheduleOption):
        if option.start_time > self.latest_start_timestep:
            raise ValueError(
                f"Option start time {option.start_time} exceeds latest_start_timestep ({self.latest_start_timestep}) for intervention {self.name}"
            )
        self.options.append(option)
        # Update the mean_risk_profile after adding a new option.
        # self.compute_mean_risk_profile() #Not efficient, compute it once everything is initialized.

    def compute_mean_risk_profile(self) -> List[float]:
        """
        Computes the overall mean risk per timestep across all schedule options,
        using only the mean risk values that are greater than 0.
        """
        if not self.options:
            self.mean_risk_profile = []
            return self.mean_risk_profile

        # Determine the maximum number of timesteps among all schedule options.
        max_timesteps = max(len(opt.mean_risk) for opt in self.options)
        overall_means = []
        for t in range(max_timesteps):
            # Gather all positive mean risk values from the t-th timestep of each option.
            timestep_values = []
            for opt in self.options:
                if t < len(opt.mean_risk) and opt.mean_risk[t] > 0:
                    timestep_values.append(opt.mean_risk[t])
            # Compute average if there are any positive values; otherwise, default to 0.
            if timestep_values:
                overall_means.append(sum(timestep_values) / len(timestep_values))
            else:
                overall_means.append(0.0)
        self.mean_risk_profile = overall_means
        return self.mean_risk_profile

    def compute_mean_workload_size(self) -> float:
        """
        Computes the mean intervention size across all schedule options,
        averaging only those options where the total_workload is not 0.
        """
        # Gather sizes from options with non-zero total_workload.
        sizes = [
            option.total_workload
            for option in self.options
            if option.total_workload > 0
        ]
        self.mean_workload_size = sum(sizes) / len(sizes) if sizes else 0.0
        return self.mean_workload_size

    def __str__(self) -> str:
        # Compute mean_risk_profile if it hasn't been computed yet.
        if not self.mean_risk_profile and self.options:
            self.compute_mean_risk_profile()

        overall_risk_str = (
            "[" + ", ".join(f"{risk:.2f}" for risk in self.mean_risk_profile) + "]"
            if self.mean_risk_profile
            else "Not computed"
        )

        options_str = (
            "\n  ".join(str(opt) for opt in self.options) if self.options else "None"
        )

        return (
            f"Intervention '{self.name}':\n"
            f"  latest_start_timestep = {self.latest_start_timestep}\n"
            f"  Options Count = {len(self.options)}\n"
            f"  Overall Mean Risk = {overall_risk_str}\n"
            f"  Schedule Options:\n  {options_str}"
        )


@dataclass
class MaintenanceSchedulingInstance:
    """
    Represents a complete instance of the maintenance scheduling problem.

    Attributes:
        T: Total number of time periods in the planning horizon.
        scenarios_number: A list specifying the number of risk scenarios for each time period.
        resources: Dictionary of Resource objects keyed by resource name.
        seasons: Dictionary of Season objects keyed by season name.
        interventions: Dictionary of Intervention objects keyed by intervention name.
        exclusions: List of mutual exclusion constraints.
    """

    T: int
    scenarios_number: List[int]
    resources: Dict[str, Resource] = field(default_factory=dict)
    seasons: Dict[str, Season] = field(default_factory=dict)
    interventions: Dict[str, Intervention] = field(default_factory=dict)
    exclusions: List[Exclusion] = field(default_factory=list)

    def add_resource(self, resource: Resource):
        self.resources[resource.name] = resource

    def add_season(self, season: Season):
        self.seasons[season.name] = season

    def add_intervention(self, intervention: Intervention):
        self.interventions[intervention.name] = intervention

    def add_exclusion(self, exclusion: Exclusion):
        self.exclusions.append(exclusion)

    def __str__(self) -> str:
        resources_str = "\n".join(str(r) for r in self.resources.values())
        seasons_str = "\n".join(str(s) for s in self.seasons.values())
        interventions_str = "\n".join(str(i) for i in self.interventions.values())
        exclusions_str = "\n".join(str(e) for e in self.exclusions)
        return (
            f"Maintenance Scheduling Instance:\n"
            f"  Planning Horizon (T): {self.T}\n"
            f"  Scenarios per period: {self.scenarios_number}\n"
            f"  Resources:\n{resources_str}\n"
            f"  Seasons:\n{seasons_str}\n"
            f"  Interventions:\n{interventions_str}\n"
            f"  Exclusions:\n{exclusions_str}"
        )

    def show(self, indent: int = 0) -> None:
        """
        Print a detailed, hierarchical view of the instance.
        If any list has more than 20 elements, only the first 20 are shown followed by an ellipsis.
        """
        pad = " " * indent
        print(f"{pad}Maintenance Scheduling Instance:")
        print(f"{pad}  Planning Horizon (T): {self.T}")
        print(f"{pad}  Scenarios per period: {trim_list(self.scenarios_number)}")

        # Resources
        print(f"{pad}  Resources:")
        if self.resources:
            for res in self.resources.values():
                print(f"{pad}    Resource: {res.name}")
                print(f"{pad}      Capacity min: {trim_list(res.capacity_min)}")
                print(f"{pad}      Capacity max: {trim_list(res.capacity_max)}")
        else:
            print(f"{pad}    None")

        # Seasons
        print(f"{pad}  Seasons:")
        if self.seasons:
            for season in self.seasons.values():
                print(f"{pad}    Season: {season.name}")
                if season.periods:
                    print(
                        f"{pad}      Periods: {trim_list(season.periods)} (range: {min(season.periods)}-{max(season.periods)})"
                    )
                else:
                    print(f"{pad}      No periods assigned")
        else:
            print(f"{pad}    None")

        # Interventions
        print(f"{pad}  Interventions:")
        if self.interventions:
            for intv in self.interventions.values():
                print(f"{pad}    Intervention: {intv.name}")
                print(f"{pad}      Latest start timestep: {intv.latest_start_timestep}")
                if intv.options:
                    print(f"{pad}      Options:")
                    for option in intv.options:
                        print(f"{pad}        Schedule Option:")
                        print(f"{pad}          Start time: {option.start_time}")
                        print(f"{pad}          Duration: {option.duration}")
                        # Workloads
                        print(f"{pad}          Workloads:")
                        if option.workloads:
                            for res_name, wl in option.workloads.items():
                                print(
                                    f"{pad}            Resource {res_name}: {trim_list(wl)}"
                                )
                        else:
                            print(f"{pad}            None")
                        # Risks
                        print(f"{pad}          Risks:")
                        if option.risks:
                            for i, risk_list in enumerate(option.risks):
                                print(
                                    f"{pad}            Timestep {i + 1}: {trim_list(risk_list)}"
                                )
                        else:
                            print(f"{pad}            None")
                else:
                    print(f"{pad}      No scheduling options available")
        else:
            print(f"{pad}    None")

        # Exclusions
        print(f"{pad}  Exclusions:")
        if self.exclusions:
            for excl in self.exclusions:
                print(f"{pad}    Exclusion Constraint:")
                print(f"{pad}      Interventions: {trim_list(excl.interventions)}")
                print(f"{pad}      Season: {excl.season}")
        else:
            print(f"{pad}    None")


# ------------------------------------------------------------------------------
# Function to load from the original JSON structure into the new object model
# ------------------------------------------------------------------------------


def load_maintenance_instance_from_json(
    json_data: Dict[str, Any], verbose: bool = False
) -> MaintenanceSchedulingInstance:
    """
    Load a MaintenanceSchedulingInstance from a JSON-like dictionary.
    This function populates the core data classes used by the repository.

    Mapping Overview:
      "T"                ⟶ MaintenanceSchedulingInstance.T
      "Scenarios_number" ⟶ MaintenanceSchedulingInstance.scenarios_number
      "Resources"        ⟶ Resources -> Resource objects
      "Seasons"          ⟶ Seasons -> Season objects
      "Interventions"    ⟶ Intervention objects with InterventionScheduleOption(s)
         • "tmax"       ⟶ Intervention.latest_start_timestep
         • "Delta"      ⟶ For each option, InterventionScheduleOption.duration (with start_time given by the option index, 1-indexed)
         • "workload"   ⟶ InterventionScheduleOption.workloads: For each resource, create a list of length T. For each absolute timestep (1 to T),
                              if the JSON specifies a workload for that timestep for the current option's start time, use it;
                              otherwise, leave it as zero.
         • "risk"       ⟶ InterventionScheduleOption.risks: For each absolute timestep (1 to T), if the JSON specifies risks for that timestep
                              for the current option's start time, use that list; otherwise, use an empty list.
      "Exclusions"       ⟶ Exclusion objects (last element is the season, preceding elements are interventions)
    """
    try:
        T = json_data["T"]
        scenarios_number = json_data["Scenarios_number"]
    except Exception as e:
        logger.error(f"Error reading instance horizon or scenarios: {e}", exc_info=True)
        T = 0
        scenarios_number = []
    instance = MaintenanceSchedulingInstance(T=T, scenarios_number=scenarios_number)
    if verbose:
        print(
            "Starting to load instance from JSON with T =",
            T,
            "and scenarios_number =",
            scenarios_number,
        )

    # Load Resources.
    if verbose:
        print("Loading resources...")
    _total_resources = len(json_data.get("Resources", {}))
    _res_counter = 0
    for res_name, res_data in json_data.get("Resources", {}).items():
        _res_counter += 1
        # print(f"Loading resource {_res_counter} of {_total_resources}: {res_name}")
        try:
            resource = Resource(
                name=res_name,
                capacity_min=res_data["min"],
                capacity_max=res_data["max"],
            )
            instance.add_resource(resource)
        except Exception as e:
            logger.error(f"Error loading resource '{res_name}': {e}", exc_info=True)
            continue
    if verbose:
        print("Finished loading resources.")

    # Load Seasons.
    if verbose:
        print("Loading seasons...")
    _total_seasons = len(json_data.get("Seasons", {}))
    _season_counter = 0
    for season_name, periods in json_data.get("Seasons", {}).items():
        _season_counter += 1
        # print(f"Loading season {_season_counter} of {_total_seasons}: {season_name}")
        try:
            season = Season(name=season_name, periods=periods)
            instance.add_season(season)
        except Exception as e:
            logger.error(f"Error loading season '{season_name}': {e}", exc_info=True)
            continue
    if json_data.get("Seasons"):
        configure_season_blocks(json_data["Seasons"])
    if verbose:
        print("Finished loading seasons.")

    # Load Interventions.
    if verbose:
        print("Loading interventions...")
    _total_interventions = len(json_data.get("Interventions", {}))
    _intv_counter = 0
    for intv_name, intv_data in json_data.get("Interventions", {}).items():
        _intv_counter += 1
        intv_name = f"I{intv_name.split('_')[1]}"
        # print(f"Loading intervention {_intv_counter} of {_total_interventions}: {intv_name}")
        try:
            latest_start_timestep = int(intv_data["tmax"])
        except Exception as e:
            logger.error(
                f"Error loading tmax for intervention '{intv_name}': {e}", exc_info=True
            )
            continue

        try:
            intervention = Intervention(
                name=intv_name,
                latest_start_timestep=latest_start_timestep,
            )
        except Exception as e:
            logger.error(
                f"Error creating intervention '{intv_name}': {e}", exc_info=True
            )
            continue

        delta_list = intv_data.get("Delta", [])
        workload_data = intv_data.get("workload", {})
        risk_data = intv_data.get("risk", {})

        _total_options = len(delta_list)
        _option_counter = 0
        # Iterate over each possible start time option (using 1-indexing).
        for i, duration in enumerate(delta_list, start=1):
            _option_counter += 1
            if i > latest_start_timestep:
                # Options beyond tmax are structurally invalid for this intervention.
                # Skip them silently to avoid polluting error logs with expected cases.
                continue
            # print(f"Loading option {_option_counter} of {_total_options} for intervention {intv_name}")
            try:
                option_workloads = {}
                for res_name, res_workload in workload_data.items():
                    # Initialize a list of zeros for absolute timesteps 1..T.
                    workload_list = [0] * instance.T
                    # For each absolute timestep from 1 to T:
                    for t in range(1, instance.T + 1):
                        # Check if this timestep is specified in the JSON.
                        if str(t) in res_workload:
                            inner_mapping = res_workload[str(t)]
                            # If the workload for this option's start time is provided, assign it.
                            if str(i) in inner_mapping:
                                workload_list[t - 1] = inner_mapping[str(i)]
                    option_workloads[res_name] = workload_list
            except Exception as e:
                logger.error(
                    f"Error loading workload for intervention '{intv_name}', option {i}: {e}",
                    exc_info=True,
                )
                continue  # Skip this option if workload processing fails.

            try:
                # Initialize risks as a list of empty lists (one per absolute timestep).
                option_risks = [[] for _ in range(instance.T)]
                for t in range(1, instance.T + 1):
                    if str(t) in risk_data:
                        inner_mapping = risk_data[str(t)]
                        if str(i) in inner_mapping:
                            option_risks[t - 1] = inner_mapping[str(i)]
            except Exception as e:
                logger.error(
                    f"Error loading risk for intervention '{intv_name}', option {i}: {e}",
                    exc_info=True,
                )
                continue  # Skip this option if risk processing fails.

            try:
                option = InterventionScheduleOption(
                    start_time=i,
                    duration=duration,
                    workloads=option_workloads,
                    risks=option_risks,
                )
                intervention.add_option(option)
            except Exception as e:
                logger.error(
                    f"Error adding schedule option for intervention '{intv_name}', option {i}: {e}",
                    exc_info=True,
                )
                continue
        # print(f"Finished loading options for intervention {intv_name}.")
        # Compute the overall mean risks list once all options are added.
        intervention.compute_mean_risk_profile()
        intervention.compute_mean_workload_size()

        try:
            instance.add_intervention(intervention)
        except Exception as e:
            logger.error(f"Error adding intervention '{intv_name}': {e}", exc_info=True)
            continue
    if verbose:
        print("Finished loading interventions.")

    # Load Exclusions.
    if verbose:
        print("Loading exclusions...")
    _total_exclusions = len(json_data.get("Exclusions", {}))
    _excl_counter = 0
    for excl_key, excl_list in json_data.get("Exclusions", {}).items():
        _excl_counter += 1
        # print(f"Loading exclusion {_excl_counter} of {_total_exclusions}: {excl_key}")
        try:
            if excl_list:
                season = excl_list[-1]
                interventions_excluded = excl_list[:-1]
                exclusion = Exclusion(
                    interventions=interventions_excluded, season=season
                )
                instance.add_exclusion(exclusion)
        except Exception as e:
            logger.error(f"Error loading exclusion '{excl_key}': {e}", exc_info=True)
            continue
    if verbose:
        print("Finished loading exclusions.")
        print("Finished loading instance.")
    return instance


@dataclass
class MaintenanceSchedule:
    """
    Represents a solution to the maintenance scheduling problem.
    """

    intervention_start_times: Dict[str, int] = field(default_factory=dict)

    def __init__(self, solution_path: str):
        """
        Initialize solution from a text file.
        """
        self.intervention_start_times = {}
        try:
            with open(solution_path, "r") as f:
                for line in f:
                    fields = line.split()
                    # Solution files have no header: every "name start" line is kept.
                    if len(fields) == 2 and fields[1].isdigit():
                        intervention, start_time = fields
                        intervention = intervention.replace("Intervention_", "I")
                        self.intervention_start_times[intervention] = int(start_time)
        except Exception as e:
            logger.error(
                f"Error loading solution from {solution_path}: {e}", exc_info=True
            )

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.intervention_start_times.items())

    def compute_active_interventions(
        self, instance: MaintenanceSchedulingInstance
    ) -> None:
        """
        Computes the list of active interventions for each timestep.
        """
        self.active_interventions_by_timestep = [[] for _ in range(int(instance.T))]
        for intervention_name, start_time in self.intervention_start_times.items():
            intervention = instance.interventions.get(intervention_name)
            if not intervention:
                continue
            matching_option = next(
                (opt for opt in intervention.options if opt.start_time == start_time),
                None,
            )
            if not matching_option:
                continue
            for t in range(start_time, start_time + matching_option.duration):
                if 1 <= t <= instance.T:
                    self.active_interventions_by_timestep[t - 1].append(
                        intervention_name
                    )
        self.active_intervention_counts = [
            len(interventions)
            for interventions in self.active_interventions_by_timestep
        ]
        self.peak_active_intervention_count = max(self.active_intervention_counts)

    def _resolve_selected_options(
        self, instance: MaintenanceSchedulingInstance
    ) -> Dict[str, InterventionScheduleOption]:
        """Cache and return the chosen schedule option for each intervention in the solution."""
        if hasattr(self, "_selected_options"):
            return self._selected_options

        option_map: Dict[str, InterventionScheduleOption] = {}
        for intv_name, start_time in self.intervention_start_times.items():
            intervention = instance.interventions.get(intv_name)
            if not intervention:
                continue
            option = next(
                (opt for opt in intervention.options if opt.start_time == start_time),
                None,
            )
            if option:
                option_map[intv_name] = option

        self._selected_options = option_map
        return option_map

    def compute_risk_profile(
        self, instance: MaintenanceSchedulingInstance
    ) -> List[float]:
        """Compute and store daily mean risk aggregated across active interventions."""
        if not hasattr(self, "active_interventions_by_timestep"):
            self.compute_active_interventions(instance)

        selected_options = self._resolve_selected_options(instance)
        T = instance.T
        risk_profile = [0.0 for _ in range(T)]

        for t in range(1, T + 1):
            active = self.active_interventions_by_timestep[t - 1]
            risk_t = 0.0
            for intv_name in active:
                option = selected_options.get(intv_name)
                if not option:
                    continue
                timestep_risks = (
                    option.risks[t - 1] if t - 1 < len(option.risks) else []
                )
                if timestep_risks:
                    risk_t += sum(timestep_risks) / len(timestep_risks)
            risk_profile[t - 1] = risk_t

        self.risk_profile = risk_profile
        return risk_profile

    def _compute_bottleneck_slack_profile(
        self, instance: MaintenanceSchedulingInstance
    ) -> List[float]:
        """Compute bottleneck slack per timestep using resource capacities."""
        selected_options = self._resolve_selected_options(instance)
        bottleneck_slack: List[float] = []

        for t in range(1, instance.T + 1):
            active = (
                self.active_interventions_by_timestep[t - 1]
                if hasattr(self, "active_interventions_by_timestep")
                else []
            )
            slacks: List[float] = []

            for res_name, res in instance.resources.items():
                usage = 0.0
                for intv in active:
                    option = selected_options.get(intv)
                    if not option:
                        continue
                    workloads = option.workloads.get(res_name, [])
                    if t - 1 < len(workloads):
                        usage += workloads[t - 1]

                cap_min = (
                    res.capacity_min[t - 1] if t - 1 < len(res.capacity_min) else 0.0
                )
                cap_max = (
                    res.capacity_max[t - 1]
                    if t - 1 < len(res.capacity_max)
                    else cap_min
                )
                denom = cap_max - cap_min

                if denom > 0:
                    slack = max(cap_max - usage, 0.0) / denom
                else:
                    slack = 0.0
                slacks.append(slack)

            bottleneck_slack.append(min(slacks) if slacks else 0.0)
        return bottleneck_slack

    def compute_risk_volatility_criterion(self) -> Dict[str, float]:
        """
        Measure the turbulence of the risk profile.
        Sum of absolute risk differences between consecutive timesteps.
        """
        if not hasattr(self, "risk_profile"):
            return {"Risk_Volatility": 0.0}

        risk_profile = np.array(self.risk_profile)
        diffs = np.abs(np.diff(risk_profile))
        total_jerk = float(np.sum(diffs))

        self.risk_volatility_criterion = {"Risk_Volatility": total_jerk}
        return self.risk_volatility_criterion

    def compute_max_critical_overlap_criterion(
        self, instance: MaintenanceSchedulingInstance
    ) -> Dict[str, float]:
        """
        Measure structural contingency risk.
        Maximum number of top critical interventions active simultaneously.
        """
        all_interventions = list(instance.interventions.values())
        for intv in all_interventions:
            if not intv.mean_risk_profile:
                intv.compute_mean_risk_profile()

        sorted_interventions = sorted(
            all_interventions,
            key=lambda x: np.mean(x.mean_risk_profile) if x.mean_risk_profile else 0.0,
            reverse=True,
        )
        top_10_count = max(1, int(len(all_interventions) * 0.10))
        critical_names = set(i.name for i in sorted_interventions[:top_10_count])

        if not hasattr(self, "active_interventions_by_timestep"):
            self.compute_active_interventions(instance)

        T = instance.T
        overlap_curve = [0] * T

        for t in range(T):
            active_intvs = self.active_interventions_by_timestep[t]
            count = sum(1 for name in active_intvs if name in critical_names)
            overlap_curve[t] = count

        max_overlap = max(overlap_curve) if overlap_curve else 0
        self.max_critical_overlap_criterion = {
            "Max_Critical_Overlap": float(max_overlap)
        }
        return self.max_critical_overlap_criterion

    def compute_summer_eco_stress_criterion(
        self,
        instance: MaintenanceSchedulingInstance,
        park_proximity_membership: "pd.Series",
    ) -> Dict[str, float]:
        """
        Calculates the environmental stress integral in summer.
        Sum of impact exceeding the historical median environmental impact.
        """
        T = instance.T
        if park_proximity_membership.empty:
            return {"Summer_Eco_Stress": 0.0}

        intervention_impacts = park_proximity_membership.to_dict()

        daily_env_impact = np.zeros(T)
        for t in range(T):
            active = self.active_interventions_by_timestep[t]
            day_impact = 0.0
            for i_name in active:
                day_impact += intervention_impacts.get(i_name, 0.0)
            daily_env_impact[t] = day_impact

        baseline = np.median(daily_env_impact)

        summer_indices = [t - 1 for t in range(1, T + 1) if timestep_is_summer(t)]

        eco_stress = 0.0
        for t_idx in summer_indices:
            excess = max(0, daily_env_impact[t_idx] - baseline)
            eco_stress += excess

        return {"Summer_Eco_Stress": float(eco_stress)}

    def compute_raw_criteria(
        self,
        instance: MaintenanceSchedulingInstance,
        ski_area_proximity_membership: "pd.Series",
        park_proximity_membership: "pd.Series",
    ) -> Dict[str, float]:
        """
        Compute the seven raw decision criteria.
        """
        if not hasattr(self, "active_interventions_by_timestep"):
            self.compute_active_interventions(instance)
        if not hasattr(self, "risk_profile"):
            self.compute_risk_profile(instance)

        selected_options = self._resolve_selected_options(instance)

        # 1. Reliability_Stress: Integral of risk > P90(Risk > 0)
        risk_profile = np.array(self.risk_profile, dtype=float)
        non_zero_risks = risk_profile[risk_profile > 0]
        if non_zero_risks.size > 0:
            safety_threshold = np.percentile(non_zero_risks, 90)
        else:
            safety_threshold = 0.0
        reliability_stress = np.sum(np.maximum(0, risk_profile - safety_threshold))

        # 2. Resource_Slack: Mean bottleneck margin over the horizon
        bottleneck_slacks = self._compute_bottleneck_slack_profile(instance)
        robustness_slack = float(np.mean(bottleneck_slacks)) if bottleneck_slacks else 0.0

        # 3. Ski_Season_Risk: Accumulated risk near ski resorts during the winter season.
        winter_indices = [
            t - 1 for t in range(1, instance.T + 1) if timestep_is_winter(t)
        ]

        ski_season_risk = 0.0
        for t_idx in winter_indices:
            if t_idx < len(self.active_interventions_by_timestep):
                active = self.active_interventions_by_timestep[t_idx]
                for i_name in active:
                    membership_raw = ski_area_proximity_membership.to_dict().get(
                        i_name, 0.0
                    )
                    membership = (
                        float(membership_raw) if membership_raw is not None else 0.0
                    )
                    if membership > 0.0:
                        opt = selected_options.get(i_name)
                        if opt and 0 <= t_idx < len(opt.risks):
                            timestep_risks = opt.risks[t_idx]
                            risk_val = (
                                sum(timestep_risks) / len(timestep_risks)
                                if timestep_risks
                                else 0.0
                            )
                            ski_season_risk += risk_val * membership

        # 4. Social_Weekend_Risk: Count of "Dangerous" (Risk > P80) interventions starting on Sat/Sun
        all_mean_risks = []
        for intv in instance.interventions.values():
            if not intv.mean_risk_profile:
                intv.compute_mean_risk_profile()
            if intv.mean_risk_profile:
                all_mean_risks.append(np.mean(intv.mean_risk_profile))

        risk_p80 = np.percentile(all_mean_risks, 80) if all_mean_risks else 0.0

        social_weekend_risk = 0.0
        for i_name, start_time in self.intervention_start_times.items():
            intv = instance.interventions.get(i_name)
            if intv:
                avg_risk = (
                    np.mean(intv.mean_risk_profile) if intv.mean_risk_profile else 0.0
                )
                if avg_risk > risk_p80:
                    if timestep_has_weekend(start_time):
                        social_weekend_risk += 1.0

        # 5. Max_Critical_Overlap
        overlap_metrics = self.compute_max_critical_overlap_criterion(instance)

        # 6. Risk_Volatility
        jerk_metrics = self.compute_risk_volatility_criterion()

        # 7. Summer_Eco_Stress
        eco_metrics = self.compute_summer_eco_stress_criterion(
            instance,
            park_proximity_membership,
        )

        self.raw_criteria_values = {
            "Reliability Stress": float(reliability_stress),
            "Resource Slack": float(robustness_slack),
            "Ski Season Risk": float(ski_season_risk),
            "Social Weekend Risk": float(social_weekend_risk),
            "Max Critical Overlap": overlap_metrics["Max_Critical_Overlap"],
            "Risk Volatility": jerk_metrics["Risk_Volatility"],
            "Summer Eco-Stress": eco_metrics["Summer_Eco_Stress"],
        }
        return self.raw_criteria_values
