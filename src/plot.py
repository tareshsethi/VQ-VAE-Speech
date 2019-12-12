import umap 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
colors = {0:'happy.wav', 1:'sad.wav', 2:'angry.wav', 3:'disgust.wav', 4:'ps.wav', 5:'fear.wav', 6:'neutral.wav'}
color_labels = ['happy', 'sad', 'angry', 'disgust', 'pleasant surprise', 'fear', 'neutral']

with open('X.pickle', 'rb') as x:
    X = np.asarray(pickle.load(x))

with open('Y.pickle', 'rb') as y:
    Y = np.asarray(pickle.load(y))

print(X.shape)
print(Y.shape)

reducer = umap.UMAP()
embedding = reducer.fit_transform(X)

print(embedding.shape)
# print([sns.color_palette()[int(x)] for x in Y.squeeze()])

fig, ax = plt.subplots()
pts = ax.scatter(embedding[:, 0], embedding[:, 1], c=[int(x) for x in Y.squeeze()], cmap='Spectral')
fig.gca().set_aspect('equal', 'datalim')
ax.set_title('UMAP projection of eval latent space', fontsize=24)
cb = fig.colorbar(pts, boundaries=np.arange(8)-0.5)
cb.set_ticks(np.arange(7))
cb.set_ticklabels(color_labels)
plt.savefig('umap.png')
