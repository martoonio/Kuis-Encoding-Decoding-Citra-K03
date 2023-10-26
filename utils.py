import base64

def replace_image(file_path, file_type):
    if file_type == 'svg':
        with open(file_path, 'r') as f:
            return f.read()
            # return base64.b64encode(f.read()).decode('utf-8')
    else:
        with open(file_path, 'rb') as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"