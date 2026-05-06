from django.shortcuts import render, redirect, get_object_or_404, redirect
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fix for main thread error in Django
from matplotlib import pyplot as plt
import io
import base64
from . import logic
from .models import Mission
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required

global username
labels = ['agri', 'barrenland', 'building', 'flooded', 'forest', 'grassland', 'mountains', 'sea', 'street', 'urban']
MODEL_PATH = "model/cnn_weights.hdf5"

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def history_view(request):
    missions = Mission.objects.all().order_by('-date_created')
    return render(request, 'history.html', {'missions': missions})

def delete_mission(request, mission_id):
    if request.method == 'POST':
        mission = get_object_or_404(Mission, id=mission_id)
        # Optional: Delete actual file if needed, but for now just DB record
        try:
             # Basic cleanup attempt if you want
             if os.path.exists(mission.image_path):
                 pass # skipping file delete logic for simplicity unless requested
        except:
             pass
        mission.delete()
        messages.success(request, "Mission record deleted.")
    return redirect('history')

def Optimize(request):
    if request.method == 'GET':
        return render(request, 'Optimize.html', {})

def OptimizeAction(request):
    if request.method == 'POST':
        # 1. Save uploaded file
        try:
            myfile = request.FILES['t1'].read()
            fname = request.FILES['t1'].name
            dir_path = "SupplyApp/static/mission_uploads/"
            
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                
            file_path = os.path.join(dir_path, fname)
            
            # Simple overwrite logic for now, or uniquify
            with open(file_path, "wb") as file:
                file.write(myfile)
            
            # Check if file was saved
            if not os.path.exists(file_path):
                 return render(request, 'index.html', {'data': "Error creating file", 'img': ''})

            # 2. Logic Integration - PRE-PROCESS (Resize NOW)
            # Use OpenCV to resize immediately so user sees the 200x200 version
            img = cv2.imread(file_path)
            if img is None:
                 return render(request, 'index.html', {'data': "Error: Failed to read image file.", 'img': ''})

            # Validation
            is_valid, msg = logic.validate_image_is_sar(img)
            if not is_valid:
                return render(request, 'index.html', {'data': f"Input Rejected: {msg}", 'img': ''})

            # Force resize to 300x300 as per user request (and logic requirement)
            img_resized = cv2.resize(img, (300, 300)) # Change to 300 as per request
            cv2.imwrite(file_path, img_resized)

            # Mission Profile Data to pass forward
            mission_type = request.POST.get('mission_type', 'standard')
            rank = request.POST.get('rank', '')
            name = request.POST.get('personnel_name', '')
            ranked_personnel = f"{rank} {name}".strip()

            context = {
                'image_url': f"mission_uploads/{fname}",
                'image_path': file_path, # Absolute path for internal use
                'mission_type': mission_type,
                'ranked_personnel': ranked_personnel
            }
            
            return render(request, 'select_points.html', context)

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return render(request, 'index.html', {'data': f"System Error: {str(e)}", 'img': ''})

    return render(request, 'index.html', {})

def PerformAnalysis(request):
    if request.method == 'POST':
        try:
            file_path = request.POST.get('image_path')
            mission_type = request.POST.get('mission_type')
            ranked_personnel = request.POST.get('ranked_personnel')
            
            # User coordinates (Pixel coordinates on 300x300 image)
            start_x = int(request.POST.get('start_x'))
            start_y = int(request.POST.get('start_y'))
            end_x = int(request.POST.get('end_x'))
            end_y = int(request.POST.get('end_y'))

            model = logic.load_prediction_model(MODEL_PATH)
            
            # Re-read the (already resized) image
            full_img = cv2.imread(file_path)
            
            # Process Patches
            # Note: logic.process_image_patches will resize again to 200x200 which is fine (idempotent)
            cost_matrix, class_matrix, conf_matrix, full_img = logic.process_image_patches(
                model, file_path, patch_size=50, mission_type=mission_type
            )
            
            if cost_matrix is None:
                 return render(request, 'index.html', {'data': "Error: Image dimensions too small for patch analysis.", 'img': ''})

            # Map Pixel Coordinates to Grid Coordinates
            # Grid size = 300 / 50 = 6x6
            patch_size = 50
            start_r = min(max(start_y // patch_size, 0), cost_matrix.shape[0]-1)
            start_c = min(max(start_x // patch_size, 0), cost_matrix.shape[1]-1)
            end_r = min(max(end_y // patch_size, 0), cost_matrix.shape[0]-1)
            end_c = min(max(end_x // patch_size, 0), cost_matrix.shape[1]-1)
            
            start = (start_r, start_c)
            end = (end_r, end_c)

            # Path Finding
            path = logic.astar_path(cost_matrix, start, end)
            
            # Visualise Map
            result_img = logic.visualize_results(full_img, class_matrix, path, patch_size=50)
            
            # Risk Analysis
            risk_status, risk_score = logic.analyze_path_risk(path, class_matrix)
            path_objects = logic.get_path_objects(path, class_matrix)
            all_objects = logic.get_all_objects(class_matrix)
            
            # Output to Terminal as requested
            print(f"\n--- MISSION REPORT ---")
            print(f"Objects on Route: {', '.join(path_objects)}")
            print(f"Global Terrain Scan: {', '.join(all_objects)}")
            print(f"----------------------\n")
            
            safety_explanation = logic.explain_route_safety(path, class_matrix)
            
            # XAI Explanation (LIME)
            lime_b64 = None
            if path:
                lime_b64 = logic.explain_critical_patch(model, full_img, class_matrix, conf_matrix, path, patch_size=50)

            # Encode Map
            retval, buffer = cv2.imencode('.png', result_img)
            img_b64 = base64.b64encode(buffer).decode()
            
            # Save Result Image for History
            fname = os.path.basename(file_path)
            result_fname = f"result_{fname}"
            dir_path = os.path.dirname(file_path)
            result_path = os.path.join(dir_path, result_fname)
            cv2.imwrite(result_path, result_img)
            
            if path:
                status = f"Route Found (Length: {len(path)}) | Safety: {risk_status} | Profile: {mission_type.upper()}"
            else:
                status = "No feasible route found due to obstacles."
            
            # Save Mission to DB
            mission = Mission.objects.create(
                user=None,
                mission_type=mission_type,
                status=status,
                risk_score=risk_score,
                image_path=f"mission_uploads/{result_fname}", # Relative to static
                ranked_personnel=ranked_personnel
            )

            context = {
                'data': status, 
                'img': img_b64,
                'lime_img': lime_b64,
                'mission_type': mission_type.title(),
                'ranked_personnel': ranked_personnel,
                'path_objects': path_objects,
                'all_objects': all_objects,
                'safety_explanation': safety_explanation
            }
            
            return render(request, 'index.html', context)
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return render(request, 'index.html', {'data': f"System Error: {str(e)}", 'img': ''})
            
    return redirect('index')

def literature_view(request):
    return render(request, 'literature.html')
