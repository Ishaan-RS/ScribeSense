from django.contrib import admin
from django.urls import path, include
from . import views
urlpatterns = [
path('',views.home,name="home"),
path('signup/',views.signup,name="signup"),
path('teachersignup',views.teachersignup,name="teachersignup"),
path('signin',views.signin,name="signin"),
path('teachersignin',views.teachersignin,name="teachersignin"),
path('signout',views.signout,name="signout"),
path('student',views.student,name="student"),
path('studentdocument',views.studentdocument,name="studentdocument"),
path('teacherdocument',views.teacherdocument,name="teacherdocument")
]