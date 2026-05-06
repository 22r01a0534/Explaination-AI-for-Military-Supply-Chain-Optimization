from django.db import models
from django.contrib.auth.models import User

class Mission(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    date_created = models.DateTimeField(auto_now_add=True)
    mission_type = models.CharField(max_length=50)
    status = models.CharField(max_length=200)
    risk_score = models.FloatField(default=0.0)
    # We will store relative paths to static/media
    image_path = models.CharField(max_length=500) 
    ranked_personnel = models.CharField(max_length=100, null=True, blank=True) 
    
    def __str__(self):
        return f"Mission {self.id}: {self.mission_type}"
