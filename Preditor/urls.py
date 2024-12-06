from django.urls import path
from .views import PredicaoDoenca

urlpatterns = [
    path("predicao/", PredicaoDoenca.as_view(), name="predicao-doenca"),
]