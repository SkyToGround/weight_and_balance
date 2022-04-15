from typing import Union, List, Tuple, Optional
import numpy as np
import pylab as pl
from enum import Enum
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import itertools
from matplotlib import ticker
from scipy.spatial import ConvexHull
from intersects import intersects
from matplotlib.widgets import Button
from copy import deepcopy
from configparser import ConfigParser
from sys import platform
import subprocess
# from descartes import PolygonPatch
# sys.path.insert(0, os.path.dirname(os.getcwd()))
# from alphashape import alphashape
# from alpha_shapes.alpha_shapes import Alpha_Shaper

inch = 2.54 #cm
pound = 0.45359 #kg
jeta1_density = 0.805 # kg/liter


class CalcFuelArm:
    def __init__(self):
        self.fuel_mass = (np.sort(np.append(np.arange(5, 332, 5), [326.7, 332]))*6.7)
        self.fuel_moment = np.array([6.8, 13.7, 20.6,
                                27.5, 34.3, 41.2,
                                48.1, 55.0, 61.8,
                                68.7, 75.6, 82.5,
                                89.3, 96.2, 103.1,
                                109.9, 116.8, 123.6,
                                130.5, 137.3, 144.2,
                                151.0, 157.9, 164.7,
                                171.6, 178.4, 185.3,
                                192.1, 198.9, 205.8,
                                212.6, 219.4, 226.3,
                                233.1,
                                239.9, 246.7, 253.5,
                                260.4, 267.2, 274.0,
                                280.8, 287.6, 294.4,
                                301.2, 308.0, 314.8,
                                321.6, 328.4, 335.2,
                                342.0, 348.8, 355.6,
                                362.4, 369.2, 376.0,
                                382.8, 389.5, 396.3,
                                403.1, 409.9, 416.7,
                                423.4, 430.2, 437.0,
                                443.7, 446.1, 450.5, 453.2
                                ])*1000
        self.arm = (self.fuel_moment / self.fuel_mass) * inch
        self.mass = self.fuel_mass*pound
        self.interp_func = interp1d(self.mass, self.arm)

    def get_arm(self, used_fuel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        used_fuel = deepcopy(used_fuel)
        if isinstance(used_fuel, np.ndarray):
            used_fuel[np.where(used_fuel < self.mass[0])] = self.mass[0]
        elif used_fuel < self.mass[0]:
            return self.arm[0]
        return self.interp_func(used_fuel)


fuel_arm_calc = CalcFuelArm()


class WBPoint:
    def __init__(self, mass: float, arm: float):
        self.mass = mass
        self.arm = arm

    def __str__(self) -> str:
        return f"{self.mass:4.0f} kg at {self.arm:4.0f} cm"


class WBLine:
    def __init__(self, points: List[WBPoint] = None):
        self._line_points: List[WBPoint] = []
        if points is not None:
            self._line_points = points

    def add_point(self, point: Union[WBPoint, List[WBPoint]]):
        if isinstance(point, list):
            self._line_points += point
        else:
            self._line_points.append(point)

    def get_wb_line(self) -> Tuple[np.ndarray, np.ndarray]:
        mass = np.empty(len(self._line_points))
        arm = np.empty(len(self._line_points))
        for i, p in enumerate(self._line_points):
            mass[i] = p.mass
            arm[i] = p.arm
        return mass, arm

    def plot_line(self, ax: pl.axis, label=None):
        mass_line, arm_line = self.get_wb_line()
        return ax.plot(arm_line, mass_line, label=label)


class LoadPoint:
    def __init__(self, arm: float, name: str = ""):
        self._arm = arm
        self.name = name

    @property
    def arm(self) -> float:
        return self._arm


class Load(LoadPoint):
    def __init__(self, mass: float, arm: float, name: str = "", lateral_pos: float = 0.0):
        super().__init__(arm, name)
        self._mass = mass
        self._lateral_pos = lateral_pos

    @property
    def lateral_pos(self) -> float:
        return self._lateral_pos

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, new_mass):
        self._mass = new_mass


class FuelLoad(Load):
    def __init__(self, max_mass: float, arm: float):
        super().__init__(0, arm)
        self._max_mass = max_mass

    @property
    def arm(self) -> float:
        return fuel_arm_calc.get_arm(self.mass)

    @property
    def max_mass(self) -> float:
        return self._max_mass

    @property
    def mass(self) -> float:
        return self._mass

    @mass.setter
    def mass(self, new_mass):
        if isinstance(new_mass, np.ndarray) and new_mass.max() > self._max_mass:
            raise RuntimeError("Fuel mass is grater than max fuel mass.")
        elif not isinstance(new_mass, np.ndarray) and new_mass > self._max_mass:
            raise RuntimeError("Fuel mass is grater than max fuel mass.")
        self._mass = new_mass


class Seat(LoadPoint):
    def __init__(self, arm: float, seat_nr: int, lateral_pos: float):
        super().__init__(arm, f"Seat {seat_nr}")
        self.lateral_pos = lateral_pos
        self.seat_nr = seat_nr

    def __eq__(self, other: "Seat"):
        return self.seat_nr == other.seat_nr and self.arm == other.arm

    def __lt__(self, other: "Seat"):
        if self.arm == other.arm:
            return self.seat_nr < other.seat_nr
        return self.arm < other.arm


class Passenger(Load):
    def __init__(self, mass: float, seat: Seat):
        super().__init__(mass, seat.arm, seat.name)
        self._seat = seat

    def __str__(self):
        return f"{self._seat}: {self.mass:0.f}kg"


class Aircraft:
    def __init__(self, empty_wb: WBPoint, pilot_arm: float, fuel_tank: FuelLoad, limits: WBLine):
        self._empty_mass = empty_wb.mass
        self._empty_arm = empty_wb.arm
        self._pilot_arm = pilot_arm
        self._pilots: List[Load] = []
        self._fuel_tank = fuel_tank
        self._loads: List[Load] = []
        self._limits = limits

    def set_pilot_mass(self, pilot_mass: float):
        if pilot_mass > 110:  # More than 110 kg == two pilots
            self._pilots = [Load(pilot_mass / 2, self._pilot_arm, lateral_pos=-42), Load(pilot_mass / 2, self._pilot_arm, lateral_pos=42)]
            return
        self._pilots = [Load(pilot_mass, self._pilot_arm, lateral_pos=-42),]

    def set_fuel_mass(self, fuel_mass: Union[float, int, np.ndarray]):
        self._fuel_tank.mass = fuel_mass

    def set_passengers(self, loads: List[Load]):
        self._loads = loads

    def add_load(self, load: Load):
        self._loads.append(load)

    def clear_loads(self):
        self._loads.clear()

    def get_weight_and_balance(self) -> Union[WBPoint, List[WBPoint]]:
        pilot_mass = np.array([m.mass for m in self._pilots])
        pilot_arms = np.array([m.arm for m in self._pilots])
        load_mass = np.array([m.mass for m in self._loads])
        load_arms = np.array([m.arm for m in self._loads])
        total_weight = self.get_total_mass()
        total_arm = ((load_arms*load_mass).sum() + (pilot_arms*pilot_mass).sum() + self._empty_mass*self._empty_arm + self._fuel_tank.mass*self._fuel_tank.arm) / total_weight
        if isinstance(total_arm, np.ndarray):
            return [WBPoint(total_weight[i], total_arm[i]) for i in range(len(total_arm))]
        return WBPoint(total_weight, total_arm)

    def get_total_mass(self) -> Union[float, np.ndarray]:
        pilot_mass = np.array([m.mass for m in self._pilots])
        load_mass = np.array([m.mass for m in self._loads])
        return pilot_mass.sum() + self._fuel_tank.mass + load_mass.sum() + self._empty_mass

    def within_limits(self, wb: Optional[Union[WBPoint, List[WBPoint]]] = None) -> bool:
        if wb is None:
            wb = self.get_weight_and_balance()
        if not isinstance(wb, List):
            wb = [wb, ]
        reference_arm = 510.0
        reference_weight = 2000.0
        test_mass, test_arm = self._limits.get_wb_line()
        for c_wb in wb:
            for i in range(len(test_mass) - 1):
                if intersects(((reference_arm, reference_weight), (c_wb.arm, c_wb.mass)), ((test_arm[i], test_mass[i]), (test_arm[i + 1], test_mass[i + 1]))):
                    return False
        return True

    @property
    def no_fuel_mass(self):
        current_total_mass = self.get_total_mass()
        return current_total_mass - self._fuel_tank.mass

    @property
    def max_fuel_mass(self):
        mass, arm = self._limits.get_wb_line()
        max_to_mass = mass.max()
        current_no_fuel_mass = self.no_fuel_mass
        remainder_mass = max_to_mass - current_no_fuel_mass
        if remainder_mass > self._fuel_tank.max_mass:
            return self._fuel_tank.max_mass
        return remainder_mass


def get_passengers(nr_of: int, min_mass: float, max_mass: float, total_mass: float) -> np.ndarray:
    assert total_mass / max_mass <= nr_of
    passengers = np.ones(nr_of)*min_mass
    diff_mass = (max_mass - min_mass)
    nr_of_heavy_passengers = int((total_mass - passengers.sum()) // diff_mass)
    passengers[0:nr_of_heavy_passengers] = min_mass + diff_mass
    remainder_mass = total_mass - passengers.sum()
    if remainder_mass > 0:
        passengers[nr_of_heavy_passengers] = min_mass + remainder_mass
    np.testing.assert_almost_equal(total_mass, passengers.sum(), decimal=2, err_msg='', verbose=True)
    return passengers


class Balance(Enum):
    FrontHeavy = 0
    BackHeavy = 1


def place_passengers(balance: Balance, seats: List[Seat], passengers: np.ndarray, shifts_back: int) -> List[Passenger]:
    seats.sort()
    passengers.sort()
    if balance == Balance.FrontHeavy:
        passengers = passengers[::-1]
    seats = seats[shifts_back:]
    return_list: List[Passenger] = []
    for i, p in enumerate(passengers):
        return_list.append(Passenger(p, seats[i]))
    return return_list


@ticker.FuncFormatter
def fuel_formatter(x, pos):
    return f"{x:.0f}"


def passengers_to_used_seats(passengers: List[Passenger]):
    return np.array([p._seat.seat_nr for p in passengers])


def seat_sub_selection(seats: List[Seat], selection: np.ndarray):
    return_list = []
    for s in seats:
        if s.seat_nr in selection:
            return_list.append(s)
    return return_list


def remove_exit_group(passengers: List[Passenger], nr_in_group: int) -> Tuple[List[Passenger], np.ndarray]:
    if nr_in_group > len(passengers):
        raise RuntimeError("Trying to remove more passengers than available.")
    return_weights = []
    for i in range(nr_in_group):
        return_weights.append(passengers.pop().mass)
    return passengers, np.array(return_weights)


def calc_exit_cb_point(plane: Aircraft, seats_balance: Balance, exit_balance: Balance, fuel_mass: Union[float, np.ndarray], seats: List[Seat], exit_seats: List[Seat], passengers: np.ndarray, passenger_shift: int, nr_passengers_left: int, nr_in_exit_grp: int):
    passengers_placed = place_passengers(balance=seats_balance, seats=seats, passengers=passengers,
                                           shifts_back=passenger_shift)[0:len(passengers) - nr_passengers_left]
    passengers_placed, exit_group = remove_exit_group(passengers_placed, nr_in_exit_grp)
    exit_grp_placed = place_passengers(balance=exit_balance, seats=exit_seats, passengers=exit_group,
                                             shifts_back=0)
    plane.set_fuel_mass(fuel_mass)
    plane.set_passengers(passengers_placed + exit_grp_placed)
    c_wb = plane.get_weight_and_balance()
    return c_wb, plane.within_limits(c_wb)

def get_setting(config: ConfigParser, config_name: str, description: str, numeric_type: type, default: Optional[int] = None):
    while True:
        try:
            print(f"{description:>30s} ({default:4.0f}):", end='')
            new_value = input()
            if new_value == "" and default is not None:
                break
            default = numeric_type(new_value)
        except ValueError as e:
            print(f"The input \"{new_value}\" can not be converted to a number. The error was: {e}")
        break
    return default

def get_float_setting(config: ConfigParser, config_name: str, description: str, default: Optional[int] = None):
    return get_setting(config, config_name, description, float, default)

def get_int_setting(config: ConfigParser, config_name: str, description: str, default: Optional[int] = None):
    return get_setting(config, config_name, description, int, default)

def main():
    config = ConfigParser()
    pilot_mass = get_float_setting(config, "pilot_mass", "Pilot mass [kg]", 185)
    max_fuel_mass_liter = get_float_setting(config, "max_fuel", "Max fuel [ℓ]", 1200)
    max_fuel_mass = max_fuel_mass_liter * jeta1_density
    jumpers = get_int_setting(config, "nr_of_skydivers", "Nr of skydivers", 13)
    jumper_total_mass = get_float_setting(config, "jumper_mass", "Skydiver total mass (w. gear) [kg]", jumpers*92)
    jumper_min_mass = get_float_setting(config, "jumper_min_mass", "Skydiver min. mass (w/o gear) [kg]", 65)
    jumper_max_mass = get_float_setting(config, "jumper_max_mass", "Skydiver max. mass (w/o gear) [kg]", 112)
    right = -42.0
    left = 42.0
    center = 0.0
    origin = 422.0
    offset = 44.0
    seats = [Seat(origin + offset * 0, 1, right),
             Seat(origin + offset * 0, 2, left),
             Seat(origin + offset * 1, 3, right),
             Seat(origin + offset * 1, 4, left),
             Seat(origin + offset * 2, 5, right),
             Seat(origin + offset * 2, 6, left),
             Seat(origin + offset * 3, 7, right),
             Seat(origin + offset * 3, 8, left),
             Seat(origin + offset * 4, 9, right),
             Seat(origin + offset * 4, 10, left),
             Seat(origin + offset * 5, 11, right),
             Seat(origin + offset * 5, 12, left),
             Seat(origin + offset * 6, 13, right),
             Seat(origin + offset * 6, 14, left),
             Seat(origin + offset * 7, 15, right),
             Seat(origin + offset * 7, 16, left),
             Seat(origin + offset * 8, 17, right),
             Seat(origin + offset * 8, 18, left),
             ]
    exit_seats = [
        Seat(265 * inch, 6, -110),
        Seat(294.5 * inch, 1, -100),
        Seat(294.5 * inch, 2, -55),
        Seat(319.5 * inch, 3, -95),
        Seat(319.5 * inch, 4, -50),
        Seat(339 * inch, 5, -95.0),
    ]
    wb_limits = WBLine([WBPoint(4500*pound, 179.6*inch), WBPoint(5500*pound, 179.6*inch), WBPoint(8000*pound, 193.37*inch), WBPoint(8750*pound, 199.15*inch), WBPoint(9062*pound, 200.23*inch), WBPoint(9062*pound, 204.35*inch), WBPoint(4500*pound, 204.35*inch)])
    lsk = Aircraft(empty_wb=WBPoint(2050.0, 468.5), pilot_arm=135.5*inch, fuel_tank=FuelLoad(max_mass=2224*pound, arm=203.3*inch), limits=wb_limits)
    lsk.set_pilot_mass(pilot_mass)
    no_fuel_no_passenger_mass = lsk.no_fuel_mass
    pilot_wb_point = lsk.get_weight_and_balance()
    passengers = get_passengers(nr_of=jumpers, min_mass=jumper_min_mass + 10, max_mass=jumper_max_mass + 10, total_mass=jumper_total_mass)
    placed_passengers = place_passengers(balance=Balance.FrontHeavy, seats=seats, passengers=passengers, shifts_back=0)
    lsk.set_passengers(placed_passengers)

    fuel_mass_range = np.linspace(0, min(lsk.max_fuel_mass - 0.1, max_fuel_mass), 10, endpoint=True)
    balance_alts = (Balance.FrontHeavy, Balance.BackHeavy)

    lsk.clear_loads()
    lsk.set_fuel_mass(fuel_mass_range)
    fuel_wb_line = WBLine()
    fuel_wb_line.add_point(lsk.get_weight_and_balance())

    shifts = len(seats) - len(passengers)
    list_of_seats_used = []
    best_seats = []
    best_exit_group = np.zeros(len(passengers))
    best_wb_points = WBLine()
    best_seated_wb_points = WBLine()
    for s in range(0, shifts + 1):
        shift_is_ok = True
        c_seated_wb_points = WBLine()
        for balance in balance_alts:
            lsk.set_fuel_mass(fuel_mass_range)
            passengers_placed = place_passengers(balance=balance, seats=seats, passengers=passengers, shifts_back=s)
            lsk.set_passengers(passengers_placed)
            wb = lsk.get_weight_and_balance()
            if not lsk.within_limits(wb):
                shift_is_ok = False
                break
            c_seated_wb_points.add_point(wb)

        if shift_is_ok:
            max_exit_group = np.zeros(len(passengers))
            list_of_wb_points = WBLine()
            for n_has_left in range(len(passengers)):
                max_n_exit = min(len(exit_seats), len(passengers) - n_has_left)
                max_has_exited = 0
                for n_exit in range(1, max_n_exit + 1):

                    had_to_break = False
                    for seated_balance, exit_balance in itertools.product(balance_alts, balance_alts):
                        c_wb_point, within_limits = calc_exit_cb_point(plane=lsk, seats_balance=seated_balance,
                                                                       exit_balance=exit_balance, fuel_mass=fuel_mass_range,
                                                                       seats=seats, exit_seats=exit_seats,
                                                                       passengers=passengers, passenger_shift=s,
                                                                       nr_passengers_left=n_has_left,
                                                                       nr_in_exit_grp=n_exit)
                        if within_limits:
                            list_of_wb_points.add_point(c_wb_point)
                            max_has_exited = n_exit
                max_exit_group[n_has_left] = max_has_exited
            if max_exit_group.sum() > best_exit_group.sum():
                best_exit_group = max_exit_group
                best_seats = passengers_placed
                best_wb_points = list_of_wb_points
                best_seated_wb_points = c_seated_wb_points
            print(f"For shift {s}, max exit groups: {max_exit_group}")

            list_of_seats_used.append(passengers_to_used_seats(passengers_placed))
    usable_seats = np.unique(list_of_seats_used)
    print(f"Usable seats ({len(usable_seats)}): {usable_seats}")

    gs = gridspec.GridSpec(nrows=14, ncols=1)
    fig = pl.figure(figsize=(7, 10))
    ax = fig.add_subplot(gs[0:7, :])
    ax.set_ylabel("Mass (kg)")
    ax.set_xlabel("Arm (cm)")
    v_offset = 85
    v_start = 4000
    h_offset = 1
    h_start = 482
    param_lines = [
        ["Maximum fuel", f"{fuel_mass_range.max():.0f} kg / {fuel_mass_range.max()/jeta1_density:.0f} ℓ / {fuel_mass_range.max()/pound:.0f} lbs"],
        ["Pilot(s)", f"{pilot_mass:.0f} kg"],
        ["# of skydivers", f"{jumpers}"],
        ["Skydivers total (w. gear)", f"{jumper_total_mass:.0f} kg"],
        ["Skydivers min (w/o gear)", f"{jumper_min_mass:.0f} kg"],
        ["Skydivers max (w/o gear)", f"{jumper_max_mass:.0f} kg"]
    ]
    ax.text(h_start, v_start + v_offset, "Input parameters", ha="center", va="bottom", weight="bold")
    for i, param in enumerate(param_lines):
        ax.text(h_start, v_start - v_offset * i, param[0], ha="right", va="bottom")
        ax.text(h_start + h_offset, v_start - v_offset * i, param[1], ha="left", va="bottom")

    def mass_to_volume(fmass):
        return (fmass - no_fuel_no_passenger_mass) / jeta1_density

    def volume_to_mass(fvol):
        return (fvol * jeta1_density) + no_fuel_no_passenger_mass

    def mass_to_lbs_fuel(fmass):
        return (fmass - no_fuel_no_passenger_mass) / pound

    def lbs_fuel_to_mass(fvol):
        return (fvol * pound) + no_fuel_no_passenger_mass

    fuel_axis = ax.secondary_yaxis("left", functions=(mass_to_volume, volume_to_mass))
    fuel_axis.set_ylabel("Fuel (ℓ)")
    fuel_axis.yaxis.set_major_formatter(fuel_formatter)
    max_litres = fuel_mass_range.max() / jeta1_density
    fuel_axis.set_yticks(np.append(np.arange(0, max_litres, 200), [max_litres, ]))
    fuel_axis.yaxis.tick_right()
    fuel_axis.yaxis.set_label_position("right")
    fuel_lbs_axis = ax.secondary_yaxis(0.12, functions=(mass_to_lbs_fuel, lbs_fuel_to_mass))
    max_pounds = fuel_mass_range.max() / pound
    fuel_lbs_axis.set_yticks(np.append(np.arange(0, max_pounds, 400), [max_pounds, ]))
    fuel_lbs_axis.set_ylabel("Fuel (lbs)")
    fuel_lbs_axis.yaxis.tick_right()
    fuel_lbs_axis.yaxis.set_label_position("right")
    fuel_lbs_axis.yaxis.set_major_formatter(fuel_formatter)

    plot_lim_mass, plot_lim_arm = wb_limits.get_wb_line()
    ax.plot(plot_lim_arm, plot_lim_mass, "-", color="k")
    arm_landing_limits = np.array([197.22, 204.35]) * inch
    weight_landing_limits = np.array([8500, 8500]) * pound
    ax.plot(arm_landing_limits, weight_landing_limits, "--", color="k")
    ax.axis([440, None, None, None])
    ax.grid()
    ax.plot(pilot_wb_point.arm, pilot_wb_point.mass, "X", label="Pilot(s)", c="k")
    fuel_wb_line.plot_line(ax, "Fuel")
    exit_mass, exit_arms = best_wb_points.get_wb_line()
    # ax.plot(exit_arms, exit_mass, ".", label="Exit points")
    points = np.stack((exit_mass, exit_arms), axis=1)
    hull = ConvexHull(points)
    # ax.plot(points[hull.vertices,1], points[hull.vertices,0], "--", lw=2)
    ax.fill(points[hull.vertices,1], points[hull.vertices,0], label="Exit W&B", c="#ff7f0e")

    seated_mass, seated_arm = best_seated_wb_points.get_wb_line()
    seated_points = np.stack((seated_arm, seated_mass), axis=1)
    seated_hull = ConvexHull(seated_points)
    ax.fill(seated_points[seated_hull.vertices, 0], seated_points[seated_hull.vertices, 1], label="Seated W&B (TO)", c="#1f77b4")
    # shaper = Alpha_Shaper(seated_points)
    # alpha_shape = shaper.get_shape(alpha=0.2)
    # ax.add_patch(PolygonPatch(alpha_shape, color='#1f77b4'))

    # ax.plot(seated_arm, seated_mass, "o", c="k")
    ax.legend()

    ax_exits = fig.add_subplot(gs[8:10, :])
    ax_exits.plot(np.arange(len(best_exit_group)), best_exit_group, "X")
    ax_exits.set_xlabel("Skydivers left in airplane")
    ax_exits.set_ylabel("Max exit grp. size")
    ax_exits.grid(True)

    ax_seats = fig.add_subplot(gs[11:13, :])
    for pilot in lsk._pilots:
        circle = pl.Circle((pilot.arm, pilot.lateral_pos), 15.5, color='k', fill=False)
        ax_seats.add_patch(circle)
        ax_seats.text(pilot.arm, pilot.lateral_pos, "P", ha="center", va="center")
    c_used_seats = passengers_to_used_seats(best_seats)
    for c_seat in seats:
        circle = pl.Circle((c_seat.arm, c_seat.lateral_pos), 15.5, color='k', fill=False)
        ax_seats.add_patch(circle)
        ax_seats.text(c_seat.arm, c_seat.lateral_pos, f"{c_seat.seat_nr}", ha="center",
                     va="center")
        if c_seat.seat_nr not in c_used_seats:
            no_place_lines_1 = np.array([-1, 1]) * 18
            no_place_lines_2 = np.array([1, -1]) * 18
            ax_seats.plot(no_place_lines_1 + c_seat.arm, no_place_lines_1 + c_seat.lateral_pos,
                         "-", color="k")
            ax_seats.plot(no_place_lines_1 + c_seat.arm, no_place_lines_2 + c_seat.lateral_pos,
                         "-", color="k")

    for c_exit in exit_seats:
        circle = pl.Circle((c_exit.arm, c_exit.lateral_pos), 15.5, color='k', fill=False)
        ax_seats.add_patch(circle)
        ax_seats.text(c_exit.arm, c_exit.lateral_pos, "E", ha="center", va="center")
    ax_seats.margins(y=0)
    ax_seats.axis([300, 900, None, None])
    ax_seats.xaxis.set_ticks([])
    ax_seats.yaxis.set_ticks([])
    ax_seats.xaxis.set_ticklabels([])
    ax_seats.yaxis.set_ticklabels([])
    ax_seats.set_title("Seats in use and exit positions")

    ax_extra = fig.add_subplot(gs[13:, :])
    ax_extra.xaxis.set_ticks([])
    ax_extra.yaxis.set_ticks([])
    ax_extra.xaxis.set_ticklabels([])
    ax_extra.yaxis.set_ticklabels([])
    ax_extra.text(0.01, 0, "ALWAYS remain at your assigned seat until the previous group has exited.", ha="left", va="bottom")
    ax_extra.axis([0, None, 0, 4])

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(len(best_exit_group) - x):d}"
    ax_exits.xaxis.set_major_formatter(major_formatter)
    ax_exits.yaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.tight_layout()
    file_name = "w_and_b_selsk.png"
    fig.savefig(file_name, dpi=300)

    axcut = pl.axes([0.9, 0.0, 0.1, 0.075])

    def on_click(event):
        if platform == "win32":
            print("Printing on Windows")
            subprocess.call(["mspaint", "/pt", file_name])
        elif platform == "darwin":
            print("Printing on Mac OS")
            subprocess.call(["lpr", file_name])
        else:
            print("Unknown platform, cant print.")

    bcut = Button(axcut, 'Print', color='red', hovercolor='green')
    bcut.on_clicked(on_click)
    pl.show()

if __name__ == "__main__":
    main()
