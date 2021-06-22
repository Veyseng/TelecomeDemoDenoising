#required Python libraries
import cv2
import numpy as np
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import scipy
from scipy import signal
from numpy.fft import fft2, ifft2
from skimage import io, img_as_float
from skimage import color, data, restoration
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
import soundfile as sf
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise
import IPython
from scipy.io import wavfile
import librosa.display
import librosa

def voiceSimultation():
	print("\nVoice denoising simulation\n")
	inputVoice = str(input("Enter the voice file path: "))
	data, rate = sf.read(inputVoice)
	data = data
	IPython.display.Audio(data=data, rate=rate)
	fig, ax = plt.subplots(figsize=(20,3))
	ax.plot(data)
	noise_len = 2 # seconds
	noise = band_limited_noise(min_freq=2000, max_freq = 12000, samples=len(data), samplerate=rate)*10
	noise_clip = noise[:rate*noise_len]
	audio_clip_band_limited = data + noise
	fig, ax = plt.subplots(figsize=(20,3))
	ax.plot(audio_clip_band_limited)
	IPython.display.Audio(data=audio_clip_band_limited, rate=rate)
	sf.write("./voice/Addnoise.wav",audio_clip_band_limited, rate)

	#Denoising process
	noise_reduced = nr.reduce_noise(audio_clip=audio_clip_band_limited, noise_clip=noise_clip, prop_decrease=1.0, verbose=True)
	fig, ax = plt.subplots(figsize=(20,3))
	ax.plot(noise_reduced)
	IPython.display.Audio(data=noise_reduced, rate=rate)
	sf.write("./voice/Reducednoise.wav",noise_reduced, rate)
	
#Median Filtering for image simulation
def medianFiltering():
	# read the image
	imagePath = str(input("Enter valid image path: "))
	image = cv2.imread(imagePath)
	# apply the 3x3 median filter on the image
	processed_image = cv2.medianBlur(image, 3)
	# display image
	cv2.imshow('Median Filter Processing', processed_image)
	cv2.imshow("Original Image", image)
	# save image to disk
	cv2.imwrite('./image/Median/processed_image.png', processed_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("Successfully execution with median filtering... \n")

#Mean filtering for image
def meanFiltering():
	# read the image
	imagePath = str(input("Enter valid image path: "))
	# read the image
	image = cv2.imread(imagePath)
	
	# apply the 3x3 mean filter on the image
	kernel = np.ones((3,3),np.float32)/9
	processed_image = cv2.filter2D(image,-1,kernel)

	# display image
	cv2.imshow('Mean Filter Processing', processed_image)
	cv2.imshow("Original Image", image)
	# save image to disk
	cv2.imwrite('./image/Mean/processed_image.png', processed_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print("Successfully execution with median filtering... \n")

def wienerFiltering():
	rng = np.random.default_rng()

	astro = color.rgb2gray(data.astronaut())
	psf = np.ones((5, 5)) / 25
	astro = conv2(astro, psf, 'same')
	astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

	deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
						sharex=True, sharey=True)

	plt.gray()

	ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
	ax[0].axis('off')
	ax[0].set_title('Data')

	ax[1].imshow(deconvolved)
	ax[1].axis('off')
	ax[1].set_title('Self tuned restoration')

	fig.tight_layout()

	plt.show()
#----Main---- 
i = 1
while i == 1:
	print('Denoising Technique: \nPress [1] Image\nPress[2] Voice Simulation\nPress[0] Exit')
	choice = int(input("Enter your Choice: "))
	if choice == 1:
		while i == 1:
			print('Denoising image using \nPress [1] MedianFiltering\nPress [2] Mean Filtering\nPress [3] Wiener Filtering \nPress[0] Exit')
			ch = int(input("Enter your choice: "))
			if ch == 1:
				medianFiltering()
				break
			elif ch == 2:
				meanFiltering()
				break
			elif ch == 3:
				wienerFiltering()
			elif ch == 0:
				break
	elif choice == 2:
		voiceSimultation()
	elif choice == 0:
		print("Thank you for using our simulation")
		break
