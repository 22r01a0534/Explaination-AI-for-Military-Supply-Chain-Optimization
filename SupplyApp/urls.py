from django.urls import path
from . import views

urlpatterns = [
    path("index.html", views.index, name="index"),
    path("", views.index, name="index"),
    path("Optimize.html", views.Optimize, name="Optimize"),
    path("OptimizeAction", views.OptimizeAction, name="OptimizeAction"),
    path("finalize_mission", views.PerformAnalysis, name="finalize_mission"),
    path("history/", views.history_view, name="history"),
    path("literature/", views.literature_view, name="literature"),
    path("delete_mission/<int:mission_id>/", views.delete_mission, name="delete_mission"),
]