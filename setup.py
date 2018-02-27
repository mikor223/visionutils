from setuptools import setup

setup(name='visionutils',
	version='0.2.0',
	description='a colection of camera and image helper functions for working machine vision',
	url='http://github.com/mchatten/visionutils',
	author='Michael Chatten',
	author_email='chattenm@gmail.com',
	install_requires = [
		'opencv-python',
		'numpy',
	],
	license='MIT',
	packages=['visionutils'],
	zip_safe=False)

