import argparse

from flask import Flask, jsonify, request
from florence2 import Florence2
from PIL import Image

app = Flask(__name__)
florence2 = Florence2()

@app.route('/api/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        task = request.form.get('task')
        prompt = request.form.get('prompt')

        image = Image.open(image_file.stream)

        result = florence2.predict(task, image=image, prompt=prompt)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_cli(task=None, image_path=None, prompt=None):
	if task and image_path:
		image = Image.open(image_path)
		result = florence2.predict(task, image=image, prompt=prompt)
		print(result)
	else:
		while True:
			user_input = input("Enter <task> <image_path> [prompt] (or type 'exit' to quit): ")
			if user_input.lower() == 'exit':
				break
			try:
				inputs = user_input.split()
				task = inputs[0]
				image_path = inputs[1]
				prompt = inputs[2] if len(inputs) > 2 else None

				image = Image.open(image_path)
				result = florence2.predict(task, image=image, prompt=prompt)
				print(result)
			except Exception as e:
				print(f"Error: {e}. Please enter a valid <task> and <image_path>.")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Run predict API server or CLI.')
	parser.add_argument('mode', choices=['api', 'cli'], help='Mode to run the application')
	parser.add_argument('task', nargs='?', help='Task prompt for CLI mode')
	parser.add_argument('image_path', nargs='?', help='Image path for CLI mode')
	parser.add_argument('prompt', nargs='?', help='Text prompt for CLI mode')
	
	args = parser.parse_args()

	if args.mode == 'api':
		app.run(host='0.0.0.0', port=5000)
	elif args.mode == 'cli':
		run_cli(args.task, args.image_path, args.prompt)
