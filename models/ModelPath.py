import os

def get_models_path():
	# 项目名称
	p_name = 'mytrans'
	# 获取当前文件的绝对路径
	p_path = os.path.abspath(os.path.dirname(__file__))
	# 通过字符串截取方式获取
	return os.path.join(p_path[:p_path.index(p_name) + len(p_name)],"models")
def get_data_path():
	# 项目名称
	p_name = 'mytrans'
	# 获取当前文件的绝对路径
	p_path = os.path.abspath(os.path.dirname(__file__))
	# 通过字符串截取方式获取
	return os.path.join(p_path[:p_path.index(p_name) + len(p_name)],"reader/datasets")
def get_t5_path():
	# 项目名称
	p_name = 'mytrans'
	# 获取当前文件的绝对路径
	p_path = os.path.abspath(os.path.dirname(__file__))
	# 通过字符串截取方式获取
	return os.path.join(p_path[:p_path.index(p_name) + len(p_name)],"models/T5")
# print(get_mypycorrector_path())
