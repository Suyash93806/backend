from django.conf.urls import url
from . import views
from django.urls import include, path

urlpatterns = [
    path('',views.register_fn),
    path('me/', views.me),
    path('login/',views.login_fn),
    path('he/',views.he),
    path('get_new/',views.get_new),
    path('texttoemo/',views.texttoemo),
    path('faketrue/',views.faketrue),
    path('posneg/',views.posneg),
] 