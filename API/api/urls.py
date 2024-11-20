from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.auth import views as auth_views
from rest_framework.permissions import AllowAny


schema_view = get_schema_view(
    openapi.Info(
        title="Your API",
        default_version='v1',
        description="Test description",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@yourapi.local"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(AllowAny,),
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('model/', include('Model.urls')),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='swagger-schema'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
