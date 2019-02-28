"""Plot training/validation loss for both 3D and 4x4."""
import cPickle as pickle
import gzip
import seaborn as sns
import matplotlib, matplotlib.pyplot as plt


plt.switch_backend('Agg')

filenames = ['auto_encoder_3D_info.pkl.gz', 'auto_encoder_4x4_info.pkl.gz']
output_files = ['3D_training', '4x4_training']

for filename, output_file in zip(filenames, output_files):
    f = gzip.open('./output/{}'.format(filename), 'rb')
    data = pickle.load(f)
    f.close()

    val_loss = data['validation loss']
    train_loss = data['training loss']

    plt.plot(val_loss, c='r', lw=1.5)
    plt.plot(train_loss, c='g', lw=1.5)
    plt.legend(['validation', 'training'], fontsize=20)
    plt.ylim(0, 0.1)
    plt.xlabel('epoch', size=20)
    plt.ylabel('loss', size=20)
    plt.tight_layout()
    plt.savefig('./output/{}.eps'.format(output_file), format='eps')
    plt.savefig('./output/{}.jpg'.format(output_file), format='jpg')
    plt.close()