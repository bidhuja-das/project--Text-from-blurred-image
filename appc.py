from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import subprocess
import shutil
import sys
import io
import threading
import time
import cv2
from scipy import signal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

app = Flask(__name__, static_folder='static')

# Configuration
UPLOAD_FOLDER = r'C:\Users\bindh\OneDrive\Desktop\final'
CRAFT_FOLDER = r"C:\Users\bindh\OneDrive\Desktop\final\craft"
MNTSR_FOLDER = r"C:\Users\bindh\OneDrive\Desktop\final\mntsr"
OCR_FOLDER = r"C:\Users\bindh\OneDrive\Desktop\final\ocr"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'craft_result'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'mntsr_result'), exist_ok=True)

# Route to serve the index.html file directly
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Route to serve static files like CSS and JS
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Save as input_image.png
    file_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')
    file.save(file_path)
    
    print(f"Image saved to {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    return jsonify({'success': True, 'message': 'Image uploaded successfully'})

@app.route('/process/craft', methods=['POST'])
def process_craft():
    try:
        input_image_path = os.path.join(UPLOAD_FOLDER, 'input_image.png')
        if not os.path.exists(input_image_path):
            return jsonify({'error': f'Input image not found at {input_image_path}'}), 404
            
        craft_script = os.path.join(CRAFT_FOLDER, 'craft_main.py')
        if not os.path.exists(craft_script):
            return jsonify({'error': f'CRAFT script not found at {craft_script}'}), 404

        # Debugging Prints
        print(f"CRAFT script path: {craft_script}")
        print(f"CRAFT folder path: {CRAFT_FOLDER}")
        print(f"Input image path: {input_image_path}")
        print(f"Input image exists: {os.path.exists(input_image_path)}")

        # Clear previous results
        craft_result_dir = os.path.join(UPLOAD_FOLDER, 'craft_result')
        for file in os.listdir(craft_result_dir):
            file_path = os.path.join(craft_result_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Use absolute paths for reliability
        result = subprocess.run(
            [sys.executable, craft_script, input_image_path, craft_result_dir],  # Pass input and output paths
            cwd=CRAFT_FOLDER,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # Add timeout of 5 minutes
        )

        print("CRAFT Output:", result.stdout)
        print("CRAFT Error:", result.stderr)

        if result.returncode != 0:
            return jsonify({'error': f'CRAFT process failed: {result.stderr}'}), 500

        # Check if output files were created
        if not os.listdir(craft_result_dir):
            return jsonify({'warning': 'CRAFT process completed but no output files found'}), 200

        # List all files in the craft_result directory
        craft_files = [f for f in os.listdir(craft_result_dir) if os.path.isfile(os.path.join(craft_result_dir, f))]
        
        return jsonify({
            'success': True, 
            'message': 'CRAFT processing complete',
            'files': craft_files  # Return list of files for debugging
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'CRAFT processing timed out after 5 minutes'}), 500
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        print(f"Traceback: {error_details}")
        return jsonify({'error': f'Error during CRAFT processing: {str(e)}'}), 500

@app.route('/process/mntsr', methods=['POST'])
def process_mntsr():
    try:
        # Check if craft results exist
        craft_result_dir = os.path.join(UPLOAD_FOLDER, 'craft_result')
        if not os.path.exists(craft_result_dir) or not os.listdir(craft_result_dir):
            return jsonify({'error': 'No CRAFT results found to process'}), 400

        # Check if the MNTSR script exists
        mntsr_script = os.path.join(r'C:\Users\bindh\OneDrive\Desktop\final\mntsr\test_mntsr_on_image.py')
        if not os.path.exists(mntsr_script):
            return jsonify({'error': f'MNTSR script not found at {mntsr_script}'}), 404

        # Ensure the output directory exists
        mntsr_result_dir = os.path.join(UPLOAD_FOLDER, 'mntsr_result')
        os.makedirs(mntsr_result_dir, exist_ok=True)
        
        # Clear previous results
        for file in os.listdir(mntsr_result_dir):
            file_path = os.path.join(mntsr_result_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Find the detected text image from CRAFT results
        detected_text_craft = None
        craft_images = [f for f in os.listdir(craft_result_dir) if f.endswith('.png')]
        
        if 'detected_text.png' in craft_images:
            detected_text_craft = os.path.join(craft_result_dir, 'detected_text.png')
        elif craft_images:
            # Use the first available image
            detected_text_craft = os.path.join(craft_result_dir, craft_images[0])
            # Copy it as detected_text.png for consistency
            shutil.copy2(
                detected_text_craft,
                os.path.join(craft_result_dir, 'detected_text.png')
            )
            detected_text_craft = os.path.join(craft_result_dir, 'detected_text.png')
            print(f"Created detected_text.png from {craft_images[0]}")
        else:
            return jsonify({'error': 'No valid images found in CRAFT results'}), 400
        
        # Make sure the detected_text.png exists
        if not os.path.exists(detected_text_craft):
            return jsonify({'error': 'Failed to create or find detected_text.png'}), 500

        # Run the MNTSR script with explicit parameters
        result = subprocess.run(
            [sys.executable, mntsr_script, detected_text_craft, mntsr_result_dir],  # Pass input and output paths
            cwd=MNTSR_FOLDER,  # Run from MNTSR folder
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            timeout=300  # Add timeout of 5 minutes
        )

        # Print the output and error for debugging
        print("MNTSR Output:", result.stdout)
        print("MNTSR Error:", result.stderr)

        # Check if the script executed successfully
        if result.returncode != 0:
            return jsonify({'error': f'MNTSR process failed: {result.stderr}'}), 500

        # Check if output files were created
        if not os.listdir(mntsr_result_dir):
            return jsonify({'warning': 'MNTSR process completed but no output files found'}), 200

        # List all files in the mntsr_result directory
        mntsr_files = [f for f in os.listdir(mntsr_result_dir) if os.path.isfile(os.path.join(mntsr_result_dir, f))]
        
        return jsonify({
            'success': True, 
            'message': 'MNTSR processing complete',
            'files': mntsr_files  # Return list of files for debugging
        })

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'MNTSR processing timed out after 5 minutes'}), 500
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        print(f"Traceback: {error_details}")
        return jsonify({'error': f'Error during MNTSR processing: {str(e)}'}), 500

# Initialize the global OCR status variable
ocr_processing_status = {'status': 'idle', 'message': 'OCR not started', 'error': None}

def run_ocr_process():
    global ocr_processing_status
    
    try:
        ocr_processing_status = {'status': 'running', 'message': 'OCR processing started', 'error': None}
        
        # First, check if the expected directories exist
        if not os.path.exists(UPLOAD_FOLDER):
            raise FileNotFoundError(f"Base directory not found: {UPLOAD_FOLDER}")
        
        # Initialize lists to store potential image paths
        potential_image_paths = []
        
        # First priority: Check if input image exists and add it first
        input_image = os.path.join(UPLOAD_FOLDER, 'input_image.png')
        if os.path.exists(input_image):
            potential_image_paths.append(input_image)
        
        # Second priority: Check mntsr_result folder for images
        mntsr_result_dir = os.path.join(UPLOAD_FOLDER, 'mntsr_result')
        if os.path.exists(mntsr_result_dir):
            mntsr_files = [f for f in os.listdir(mntsr_result_dir) if f.endswith('.png')]
            # Priority order for MNTSR files
            priority_mntsr_filenames = [
                'enhanced_image.png',
                'deblurred_detected_text.png'
            ]
            # Sort MNTSR files by priority
            sorted_mntsr_files = sorted(mntsr_files, 
                key=lambda f: priority_mntsr_filenames.index(f) if f in priority_mntsr_filenames 
                else len(priority_mntsr_filenames))
            
            # Add MNTSR result images to potential paths
            for file in sorted_mntsr_files:
                potential_image_paths.append(os.path.join(mntsr_result_dir, file))
        
        # Third priority: Check CRAFT results
        craft_result_dir = os.path.join(UPLOAD_FOLDER, 'craft_result')
        if os.path.exists(craft_result_dir):
            craft_files = [f for f in os.listdir(craft_result_dir) if f.endswith('.png')]
            # Priority for CRAFT files
            if 'detected_text.png' in craft_files:
                potential_image_paths.append(os.path.join(craft_result_dir, 'detected_text.png'))
            # Add other CRAFT result images
            for file in craft_files:
                if file != 'detected_text.png':  # Skip if already added
                    potential_image_paths.append(os.path.join(craft_result_dir, file))
        
        # Print the potential image paths for debugging
        print(f"Potential OCR image paths (in order of priority): {potential_image_paths}")
        
        # Try to process each image until one succeeds
        success = False
        for image_path in potential_image_paths:
            if os.path.exists(image_path):
                try:
                    print(f"Attempting OCR on: {image_path}")
                    
                    # Create a failsafe copy in case the file is locked
                    failsafe_copy = os.path.join(UPLOAD_FOLDER, 'ocr_source_image.png')
                    shutil.copy2(image_path, failsafe_copy)
                    
                    # Use the failsafe copy for OCR
                    image_path = failsafe_copy
                    
                    # Load EasyOCR
                    import easyocr
                    reader = easyocr.Reader(['en'], gpu=False, quantize=True)
                    
                    # Run OCR with optimized settings
                    result = reader.readtext(
                        image_path,
                        detail=0,  # Only return text strings
                        paragraph=True,  # Group text into paragraphs
                        batch_size=4,  # Smaller batch size for memory efficiency
                    )
                    
                    # Check if any text was detected
                    if not result:
                        print(f"No text detected in {image_path}, trying next image if available")
                        continue
                    
                    # Write results to output file
                    output_text_file = os.path.join(UPLOAD_FOLDER, 'text.txt')
                    with open(output_text_file, 'w', encoding='utf-8') as f:
                        if result:
                            for text in result:
                                f.write(f"{text}\n")
                        else:
                            f.write("No text detected in the image.\n")
                    
                    print(f"OCR successful on {image_path}. Output saved to {output_text_file}")
                    success = True
                    break
                except Exception as e:
                    print(f"OCR failed on {image_path}: {str(e)}")
                    continue
        
        if not success:
            # If all attempts failed, create an empty text file
            output_text_file = os.path.join(UPLOAD_FOLDER, 'text.txt')
            with open(output_text_file, 'w', encoding='utf-8') as f:
                f.write("OCR processing failed. Could not extract text from any available image.\n")
            
            raise Exception("No valid OCR source images found or all OCR attempts failed")
        
        ocr_processing_status = {'status': 'completed', 'message': 'OCR processing complete', 'error': None}
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"OCR Exception: {str(e)}")
        print(f"OCR Traceback: {error_details}")
        ocr_processing_status = {'status': 'failed', 'message': f'OCR processing failed: {str(e)}', 'error': str(e)}
        
        # Create a failure text file
        output_text_file = os.path.join(UPLOAD_FOLDER, 'text.txt')
        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(f"OCR processing failed: {str(e)}\n")
            f.write("Please try again with a clearer image.\n")
        
@app.route('/process/ocr', methods=['POST'])
def process_ocr():
    global ocr_processing_status
    
    # Reset status
    ocr_processing_status = {'status': 'starting', 'message': 'Starting OCR process', 'error': None}
    
    # Start OCR processing in a separate thread
    ocr_thread = threading.Thread(target=run_ocr_process)
    ocr_thread.daemon = True  # Make thread exit when main thread exits
    ocr_thread.start()
    
    # Return immediately, client will poll for status
    return jsonify({
        'success': True, 
        'message': 'OCR processing started',
        'status': 'processing'
    })

@app.route('/ocr/status', methods=['GET'])
def ocr_status():
    global ocr_processing_status
    
    return jsonify({
        'status': ocr_processing_status['status'],
        'message': ocr_processing_status['message'],
        'error': ocr_processing_status['error']
    })

@app.route('/results')
def results_page():
    """Render the results page"""
    # Get list of files in the CRAFT result directory
    craft_result_dir = os.path.join(UPLOAD_FOLDER, 'craft_result')
    craft_files = []
    if os.path.exists(craft_result_dir):
        craft_files = [f for f in os.listdir(craft_result_dir) if f.endswith('.png')]
    
    # Get list of files in the MNTSR result directory
    mntsr_result_dir = os.path.join(UPLOAD_FOLDER, 'mntsr_result')
    mntsr_files = []
    if os.path.exists(mntsr_result_dir):
        mntsr_files = [f for f in os.listdir(mntsr_result_dir) if f.endswith('.png')]
    
    # Check if text.txt exists
    text_file = os.path.join(UPLOAD_FOLDER, 'text.txt')
    text_content = ""
    if os.path.exists(text_file):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except Exception as e:
            text_content = f"Error reading text file: {str(e)}"
    
    return render_template('results.html', 
                           craft_files=craft_files,
                           mntsr_files=mntsr_files,
                           text_content=text_content)

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve files from the upload folder"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/craft_result/<path:filename>')
def serve_craft_results(filename):
    """Serve files from the craft_result folder"""
    craft_result_dir = os.path.join(UPLOAD_FOLDER, 'craft_result')
    return send_from_directory(craft_result_dir, filename)

@app.route('/results/mntsr_result/<path:filename>')
def serve_mntsr_results(filename):
    """Serve files from the mntsr_result folder"""
    mntsr_result_dir = os.path.join(UPLOAD_FOLDER, 'mntsr_result')
    return send_from_directory(mntsr_result_dir, filename)
@app.route('/debug/file-check', methods=['GET'])
def debug_file_check():
    """Debug endpoint to check if files exist in the expected locations."""
    results = {}
    
    # Check upload folder
    results['upload_folder'] = {
        'path': UPLOAD_FOLDER,
        'exists': os.path.exists(UPLOAD_FOLDER),
        'is_dir': os.path.isdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else False,
        'files': os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) and os.path.isdir(UPLOAD_FOLDER) else []
    }
    
    # Check craft_result folder
    craft_result_dir = os.path.join(UPLOAD_FOLDER, 'craft_result')
    results['craft_result'] = {
        'path': craft_result_dir,
        'exists': os.path.exists(craft_result_dir),
        'is_dir': os.path.isdir(craft_result_dir) if os.path.exists(craft_result_dir) else False,
        'files': os.listdir(craft_result_dir) if os.path.exists(craft_result_dir) and os.path.isdir(craft_result_dir) else []
    }
    
    # Check mntsr_result folder
    mntsr_result_dir = os.path.join(UPLOAD_FOLDER, 'mntsr_result')
    results['mntsr_result'] = {
        'path': mntsr_result_dir,
        'exists': os.path.exists(mntsr_result_dir),
        'is_dir': os.path.isdir(mntsr_result_dir) if os.path.exists(mntsr_result_dir) else False,
        'files': os.listdir(mntsr_result_dir) if os.path.exists(mntsr_result_dir) and os.path.isdir(mntsr_result_dir) else []
    }
    
    # Check specific files
    specific_files = [
        os.path.join(UPLOAD_FOLDER, 'input_image.png'),
        os.path.join(craft_result_dir, 'detected_text.png'),
        os.path.join(mntsr_result_dir, 'enhanced_image.png'),
        os.path.join(mntsr_result_dir, 'deblurred_detected_text.png')
    ]
    
    results['specific_files'] = {}
    for file_path in specific_files:
        results['specific_files'][file_path] = {
            'exists': os.path.exists(file_path),
            'is_file': os.path.isfile(file_path) if os.path.exists(file_path) else False,
            'size': os.path.getsize(file_path) if os.path.exists(file_path) and os.path.isfile(file_path) else 0
        }
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True) # Enable threading
    


