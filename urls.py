from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('w_and_b_plot/<int:skydivers>/<int:pilot_mass>/<int:total_mass>/<int:min_mass>/<int:max_mass>/<int:max_fuel_mass>/', views.generate_w_and_b_plot, name='generate_w_and_b_plot'),
]