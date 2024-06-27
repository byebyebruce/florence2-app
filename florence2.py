import time

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
TASKS = [
	"CAPTION", # 图像描述
	"DETAILED_CAPTION", # 详细图像描述
	"MORE_DETAILED_CAPTION", # 更详细图像描述
	"CAPTION_TO_PHRASE_GROUNDING", # 图像描述到定语
	"DENSE_REGION_CAPTION", # 密集区域描述
	"OD", # 目标检测
	"REGION_PROPOSAL", # 候选区域
	"OCR", # 字符识别
	"OCR_WITH_REGION", # 区域字符识别
	"REFERRING_EXPRESSION_SEGMENTATION", # REFERRING_EXPRESSION_SEGMENTATION
	"REGION_TO_SEGMENTATION", # 区域分割
	"OPEN_VOCABULARY_DETECTION", # 开放词汇下的目标检测”
	"REGION_TO_CATEGORY", # 区域类别
	"REGION_TO_DESCRIPTION" # 区域描述
]

class Florence2:
	def __init__(self, model_id='microsoft/Florence-2-large'):
		self.model_id = model_id
		self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
		self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
		self.cuda = torch.cuda.is_available()
		if self.cuda:
			self.model = self.model.cuda()	
			# 检查模型是否在 GPU 上
			print("Model is on GPU:", next(self.model.parameters()).is_cuda)
		else:
			print("CUDA is not available. Model is on CPU.")

		

	def predict(self, task, image, prompt=None):
		"""
        调用 Microsoft Florence2 模型并测量耗时

        参数:
        - task (str): 任务提示符，必须是 TASKS 中的一个值。
        - image (PIL.Image): 要处理的图像。
        - prompt (str, 可选): 附加的文本输入，可以是 None。

        返回:
        - dict: 包含模型生成的结果。

        异常:
        - ValueError: 如果 task 无效，则抛出此异常并提示有效的任务提示符。
		"""
		if task not in TASKS:
			valid_tasks = '\n'.join(TASKS)
			raise ValueError(f"Invalid task prompt: {task}. Valid task prompts are: {valid_tasks}")

		
		task_label = "<"+task+">"
		start_time = time.time()

		if prompt is None:
			prompt = task_label
		else:
			prompt = task_label + prompt

		inputs = self.processor(text=prompt, images=image, return_tensors="pt")
		input_ids = inputs["input_ids"]
		pixel_values = inputs["pixel_values"]
		if self.cuda:
			input_ids = input_ids.cuda()
			pixel_values = pixel_values.cuda()

		generated_ids = self.model.generate(
			input_ids=input_ids,
			pixel_values=pixel_values,
			max_new_tokens=1024,
			early_stopping=False,
			do_sample=False,
			num_beams=3,
		)
		generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
		parsed_answer = self.processor.post_process_generation(
			generated_text, 
			task=task_label, 
			image_size=(image.width, image.height)
		)

		end_time = time.time()
		elapsed_time = end_time - start_time
		print(f"Predict function took {elapsed_time:.2f} seconds.")

		return {'task':task, 'data':parsed_answer[task_label]}
