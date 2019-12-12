import pickle
import numpy as np

with open('train.pickle', 'rb') as input_file:
    object_dict = pickle.load(input_file)

wav_names = object_dict.keys()
X = np.zeros((2800, 1536))
Y = np.zeros((2800, 1))
names = {'happy.wav':0, 'sad.wav':1, 'angry.wav':2, 'disgust.wav':3, 'ps.wav':4, 'fear.wav':5, 'neutral.wav':6}

for i, wav in enumerate(wav_names):
    # print(object_dict[wav]['quantized'].squeeze().reshape(1, -1).shape)
    for name in names: 
        if name in wav.split('_'):
            Y[i] = names[name]
            # print (object_dict[wav]['quantized'].squeeze().flatten().shape)
            X[i] = object_dict[wav]['quantized'].squeeze().flatten().cpu().detach().numpy()
            break

print (X.shape)
Y = np.asarray(Y)
print(Y.shape)

with open('X.pickle', 'wb') as output:
    pickle.dump(X, output)

with open('Y.pickle', 'wb') as output:
    pickle.dump(Y, output)

