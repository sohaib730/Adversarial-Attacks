""""
    Author = Sohaib kiani
    Description: This is an example code to get adversarial examples from WhiteBox(CW based l2,l0) and BlackBox (PGD,Momentum Based attacks)
"""

import tensorflow as tf
import numpy as np
import time
from PIL import Image
import pickle


from setup_cifar import CIFAR, CIFARModel
#from setup_mnist import MNIST, MNISTModel
#from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


from SohaibAttacks import BlackBoxMomentum
from PGD_attack import PGDAttack

def generate_validation():
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs, labels = pickle.load(open('preprocess_validation.p', mode='rb'))
    inputs=inputs - 0.5
    return inputs, labels,labels

def generate_train_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    true_label=[]
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.train_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.train_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.train_data[start+i])
                targets.append(np.eye(data.train_labels.shape[1])[j])
                true_label.append(data.train_labels[start+i])
        else:
            inputs.append(data.train_data[start+i])
            targets.append(data.train_labels[start+i])
            true_label.append(data.train_labels[start+i])
        #print ('orig label',data.test_labels[start+i])
    inputs = np.array(inputs)
    #print (inputs)
    targets = np.array(targets)
    #print (targets)
    true_label=np.array(true_label)
    #print('true_label',true_label)
    return inputs, targets,true_label


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    true_label=[]
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                true_label.append(data.test_labels[start+i])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])
            true_label.append(data.test_labels[start+i])
        #print ('orig label',data.test_labels[start+i])
    inputs = np.array(inputs)
    #print (inputs)
    targets = np.array(targets)
    #print (targets)
    true_label=np.array(true_label)
    #print('true_label',true_label)
    return inputs, targets,true_label
def normalize(x):
    """
        argument
            - x: input image data in numpy array [32, 32, 3]
        return
            - normalized x
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

if __name__ == "__main__":
    with tf.Session() as sess:
        ##Choose Different DataSet
        #data, model =  MNIST(), MNISTModel("models/mnist", sess)
        data, model =  CIFAR(), CIFARModel("models/cifar", sess)

        #Choose Attack Method
        #attack = CarliniL2(sess, model,targeted=False, batch_size=200, max_iterations=1000, confidence=0)
        #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)
        attack=BlackBoxMomentum(sess, model, batch_size=100, max_iterations=300,max_epsilon=10)

        #attack=PGDAttack(sess, model, batch_size=1000)
        #for BlackBox batch_size and samples must be equal and trageted should be false

        print (data.train_data.shape)
        #inputs, targets, true_labels = generate_data(data, samples=32, targeted=False,start=0, inception=False)
        #inputs, targets, true_labels = generate_data(data, samples=data.test_data.shape[0], targeted=False,start=0, inception=False)


        inputs, targets, true_labels = generate_validation()
        #print ("Label Shape:",np.shape(np.argmax(targets,1)))

        #targets=np.argmax(targets,1)   ###  For PGD_Attack
        ###IF sample is 1 it will give data of shape (9,32,32,3) for targeted attack
        print (inputs.shape)
        timestart = time.time()
        adv = attack.attack(inputs, targets)  #Replace with targets for other attacks
        print (np.max(adv))
        """for i in range(len(adv)):
          adv[i]=normalize(adv[i] * 255.0)
        print (np.max(adv))
        print (np.min(adv))
        adv=adv-0.5"""
        print (adv.shape)
        timeend = time.time()

        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")
        #rescaled=((adv+0.5)*255).astype(np.uint8)
        #inputs=normalize(adv)
        #pickle.dump((inputs+0.5, true_labels), open('preprocess_batch_adv_1.p', 'wb'))
        #print (np.shape(pgd_test_adv))


        """resized_adv=[]
        resized_valid=[]
        for i in range(len(adv)):
        	image=((adv[i]+0.5)*255).astype(np.uint8)
        	img=Image.fromarray(image)
        	img=img.resize((16,16),Image.ANTIALIAS)
        	np_im = np.asarray(np.array(img))
        	np_im = np.reshape(np_im , (1,16,16,3))
        	np_im=normalize(np_im)
        	if i==0 :
        		resized_adv=np_im
        		i=i+1
        		continue
        	resized_adv=np.append(resized_adv,np_im,axis=0)

        pickle.dump((resized_adv,true_labels),open('test_resized_MI_adv.p','wb'))"""


        adv_lab= []
        for i in range(len(adv)):
                label=model.model.predict(adv[i:i+1])
                if i==0:
                    adv_lab=label
                    i=i+1
                    print ("got")
                    continue
                adv_lab=np.append(adv_lab,label,axis=0)

        pickle.dump((adv+0.5, np.argmax(true_labels,axis=1),np.argmax(adv_lab,axis=1)), open('data_adv_train_MI_ep10/preprocess_validation_adv_tal.p', 'wb'))

        for i in range(0,10):

        	image=((inputs[i]+0.5)*255).astype(np.uint8)
        	img=Image.fromarray(image)
        	img.save("output_pics/original"+str(i)+".jpeg")
        	print("Adversarial:")
        	rescaled=((adv[i]+0.5)*255).astype(np.uint8)
        	#rescaled=rescaled.reshape(28,28)
        	#print(rescaled.shape)
        	im = Image.fromarray(rescaled)
        	im.save("output_pics/adversarial"+str(i)+".jpeg")

        	#rescaled=((resized_adv[i])*255).astype(np.uint8)
        	#rescaled=rescaled.reshape(28,28)
        	#print(rescaled.shape)
        	#im = Image.fromarray(rescaled)
        	#im.save("output_pics/resized_adversarial"+str(i)+".jpeg")
        	print("Valid Classification:", np.argmax(model.model.predict(inputs[i:i+1])))
        	print("Adv Classification:", np.argmax(adv_lab[i]))



        	"""rescaled=((inputs[i]+0.5)*255).astype(np.uint8)
        	im = Image.fromarray(rescaled)
        	img=img.resize((16,16),Image.ANTIALIAS)
        	np_im = np.asarray(np.array(img))
        	np_im = np.reshape(np_im , (1,16,16,3))
        	np_im=normalize(np_im)
        	print("Total Original distortion:", np.sum((adv[i]-inputs[i])**2))
        	print("Total Adversarial distortion:", np.sum((resized_adv[i]-np_im)**2))"""
