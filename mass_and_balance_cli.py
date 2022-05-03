from typing import Optional
from matplotlib.widgets import Button
from configparser import ConfigParser
from sys import platform
import subprocess
from mass_and_balance import calc_w_and_b
import pylab as pl

jeta1_density = 0.805 # kg/liter


def get_setting(config: ConfigParser, config_name: str, description: str, numeric_type: type, default: Optional[int] = None):
    config_section = "WeightAndBalance"
    if config_section not in config:
        config.add_section(config_section)
    if config_name not in config[config_section]:
        config[config_section][config_name] = f"{default}"
    default = numeric_type(config[config_section][config_name])
    new_value = ""
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
    config[config_section][config_name] = f"{default}"
    return default

def get_float_setting(config: ConfigParser, config_name: str, description: str, default: Optional[int] = None):
    return get_setting(config, config_name, description, float, default)

def get_int_setting(config: ConfigParser, config_name: str, description: str, default: Optional[int] = None):
    return get_setting(config, config_name, description, int, default)

def calc_w_and_b_local(pilot_weight: float, max_fuel_mass: float, nr_of_skydivers: int, jumper_total_weight: float, jumper_min_weight: float, jumper_max_weight: float):
    fig = pl.figure(figsize=(7, 12))
    
    calc_w_and_b(pilot_weight=pilot_weight, nr_of_skydivers=nr_of_skydivers, jumper_total_weight=jumper_total_weight,
                 jumper_min_weight=jumper_min_weight, jumper_max_weight=jumper_max_weight, max_fuel_mass=max_fuel_mass, fig=fig)
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

def main():
    config_file_name = "config.ini"
    config = ConfigParser()
    config.read(config_file_name)
    while True:
        try:
            pilot_mass = get_float_setting(config, "pilot_mass", "Pilot mass [kg]", 185)
            max_fuel_mass_liter = get_float_setting(config, "max_fuel", "Max fuel [ℓ]", 1200)
            max_fuel_mass = max_fuel_mass_liter * jeta1_density
            jumpers = get_int_setting(config, "nr_of_skydivers", "Nr of skydivers", 13)
            jumper_total_mass = get_float_setting(config, "jumper_mass", "Skydiver total mass (w. gear) [kg]", jumpers*92)
            jumper_min_mass = get_float_setting(config, "jumper_min_mass", "Skydiver min. mass (w/o gear) [kg]", 65)
            jumper_max_mass = get_float_setting(config, "jumper_max_mass", "Skydiver max. mass (w/o gear) [kg]", 112)

            calc_w_and_b_local(pilot_mass, max_fuel_mass, jumpers, jumper_total_mass, jumper_min_mass, jumper_max_mass)
            break
        except RuntimeError as e:
            print(e)
            print("Failed to find solution. Try again.")
    config_out_file = open(config_file_name, "w")
    config.write(fp=config_out_file)


if __name__ == "__main__":
    main()
