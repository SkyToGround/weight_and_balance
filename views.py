from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib.figure import Figure
from io import BytesIO
import numpy as np
import PIL, PIL.Image
from weight_and_balance.mass_and_balance import calc_w_and_b



def generate_w_and_b_plot(request, skydivers: int, pilot_mass: int, total_mass: int, min_mass: int, max_mass: int, max_fuel_mass: int):
    fig = Figure(figsize=(7, 12), dpi=300)
    canvas = Canvas(fig)
    calc_w_and_b(pilot_weight=pilot_mass, nr_of_skydivers=skydivers, jumper_total_weight=total_mass,
                 jumper_min_weight=min_mass, jumper_max_weight=max_mass, max_fuel_mass=max_fuel_mass, fig=fig, solution_exception=False)
    buffer = BytesIO()
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")

    return HttpResponse(buffer.getvalue(), content_type="image/png")


def index(request):
    template = loader.get_template('w_and_b.html')
    context = {}
    return HttpResponse(template.render(context, request))
