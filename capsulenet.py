"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

import os


K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    PATH=os.getcwd()
    data_path=PATH + '/dataset'
    #data_path='/home/gaurav/Desktop/semproj/Vector-Capsule-Network-for-wild-animal-species-recognition-in-Camera-trap-images.-master/images'
    data_dir_list=sorted(os.listdir(data_path))
    img_data_list=[]
    for dataset in data_dir_list:
        
        img_list=sorted(os.listdir(data_path+'/'+dataset))
        print(img_list)
    #      print("loaded the images of dataset -"+"{}\n".format((dataset))
        for img in img_list:
            print(data_path+'/'+dataset+'/'+img)
            input_img=cv2.imread(data_path+'/'+dataset+'/'+img)
            if input_img is not None:             
                input_img_resize=cv2.resize(input_img,(128,128))
                img_data_list.append(input_img_resize)        
            else:                
                print("image not loaded")                                
        img_data=np.array(img_data_list) # Convert to numpy array
        img_data=img_data.astype('float32') # Convert to float
        img_data /= 255 #Normalization 
        mean = np.array([0.485, 0.456, 0.406]) #provided mean
        std = np.array([0.229, 0.224, 0.225]) #provided std
        img_data = (img_data-mean)/std

        print(img_data.shape)
        
    num_of_samples =img_data.shape[0]
    
    # Assign lables to the dataset
    labels=np.ones((num_of_samples,),dtype='int64')
    labels[0: 60] = 0
    labels[60: 120] = 1
    labels[120: 178] = 2
    labels[178: 238] = 3
    labels[238: 282] = 4
    labels[282: 323] = 5
    labels[323: 376] = 6
    labels[376: 424] = 7
    labels[424: 483] = 8
    labels[483: 543] = 9
    labels[543: 603] = 10
    labels[603: 659] = 11
    labels[659: 719] = 12
    labels[719: 779] = 13
    labels[779: 837] = 14
    labels[837: 895] = 15
    labels[895: 952] = 16
    labels[952: 997] = 17
    labels[997: 1056] = 18
    labels[1056: 1115] = 19
    labels[1115: 1175] = 20
    labels[1175: 1231] = 21
    labels[1231: 1290] = 22
    labels[1290: 1342] = 23
    labels[1342: 1402] = 24
    labels[1402: 1462] = 25
    labels[1462: 1522] = 26
    labels[1522: 1581] = 27
    labels[1581: 1641] = 28
    labels[1641: 1701] = 29
    labels[1701: 1761] = 30
    labels[1761: 1814] = 31
    labels[1814: 1873] = 32
    labels[1873: 1932] = 33
    labels[1932: 1992] = 34
    labels[1992: 2052] = 35
    labels[2052: 2111] = 36
    labels[2111: 2171] = 37
    labels[2171: 2230] = 38
    labels[2230: 2290] = 39
    labels[2290: 2350] = 40
    labels[2350: 2410] = 41
    labels[2410: 2469] = 42
    labels[2469: 2529] = 43
    labels[2529: 2589] = 44
    labels[2589: 2649] = 45
    labels[2649: 2709] = 46
    labels[2709: 2769] = 47
    labels[2769: 2829] = 48
    labels[2829: 2889] = 49
    labels[2889: 2949] = 50
    labels[2949: 3009] = 51
    labels[3009: 3069] = 52
    labels[3069: 3129] = 53
    labels[3129: 3189] = 54
    labels[3189: 3249] = 55
    labels[3249: 3309] = 56
    labels[3309: 3367] = 57
    labels[3367: 3427] = 58
    labels[3427: 3486] = 59
    labels[3486: 3546] = 60
    labels[3546: 3606] = 61
    labels[3606: 3666] = 62
    labels[3666: 3726] = 63
    labels[3726: 3776] = 64
    labels[3776: 3836] = 65
    labels[3836: 3896] = 66
    labels[3896: 3956] = 67
    labels[3956: 4016] = 68
    labels[4016: 4076] = 69
    labels[4076: 4136] = 70
    labels[4136: 4196] = 71
    labels[4196: 4256] = 72
    labels[4256: 4316] = 73
    labels[4316: 4373] = 74
    labels[4373: 4433] = 75
    labels[4433: 4493] = 76
    labels[4493: 4552] = 77
    labels[4552: 4612] = 78
    labels[4612: 4672] = 79
    labels[4672: 4732] = 80
    labels[4732: 4792] = 81
    labels[4792: 4852] = 82
    labels[4852: 4905] = 83
    labels[4905: 4965] = 84
    labels[4965: 5025] = 85
    labels[5025: 5085] = 86
    labels[5085: 5145] = 87
    labels[5145: 5205] = 88
    labels[5205: 5265] = 89
    labels[5265: 5325] = 90
    labels[5325: 5385] = 91
    labels[5385: 5445] = 92
    labels[5445: 5505] = 93
    labels[5505: 5565] = 94
    labels[5565: 5625] = 95
    labels[5625: 5684] = 96
    labels[5684: 5744] = 97
    labels[5744: 5804] = 98
    labels[5804: 5864] = 99
    labels[5864: 5914] = 100
    labels[5914: 5974] = 101
    labels[5974: 6034] = 102
    labels[6034: 6094] = 103
    labels[6094: 6143] = 104
    labels[6143: 6203] = 105
    labels[6203: 6262] = 106
    labels[6262: 6322] = 107
    labels[6322: 6382] = 108
    labels[6382: 6442] = 109
    labels[6442: 6502] = 110
    labels[6502: 6562] = 111
    labels[6562: 6612] = 112
    labels[6612: 6672] = 113
    labels[6672: 6731] = 114
    labels[6731: 6791] = 115
    labels[6791: 6850] = 116
    labels[6850: 6910] = 117
    labels[6910: 6969] = 118
    labels[6969: 7029] = 119
    labels[7029: 7089] = 120
    labels[7089: 7149] = 121
    labels[7149: 7209] = 122
    labels[7209: 7268] = 123
    labels[7268: 7327] = 124
    labels[7327: 7386] = 125
    labels[7386: 7446] = 126
    labels[7446: 7506] = 127
    labels[7506: 7566] = 128
    labels[7566: 7626] = 129
    labels[7626: 7686] = 130
    labels[7686: 7746] = 131
    labels[7746: 7806] = 132
    labels[7806: 7866] = 133
    labels[7866: 7925] = 134
    labels[7925: 7985] = 135
    labels[7985: 8045] = 136
    labels[8045: 8105] = 137
    labels[8105: 8165] = 138
    labels[8165: 8225] = 139
    labels[8225: 8283] = 140
    labels[8283: 8343] = 141
    labels[8343: 8403] = 142
    labels[8403: 8463] = 143
    labels[8463: 8523] = 144
    labels[8523: 8583] = 145
    labels[8583: 8643] = 146
    labels[8643: 8703] = 147
    labels[8703: 8762] = 148
    labels[8762: 8822] = 149
    labels[8822: 8873] = 150
    labels[8873: 8933] = 151
    labels[8933: 8992] = 152
    labels[8992: 9052] = 153
    labels[9052: 9112] = 154
    labels[9112: 9172] = 155
    labels[9172: 9231] = 156
    labels[9231: 9291] = 157
    labels[9291: 9351] = 158
    labels[9351: 9410] = 159
    labels[9410: 9470] = 160
    labels[9470: 9530] = 161
    labels[9530: 9590] = 162
    labels[9590: 9650] = 163
    labels[9650: 9710] = 164
    labels[9710: 9769] = 165
    labels[9769: 9829] = 166
    labels[9829: 9888] = 167
    labels[9888: 9947] = 168
    labels[9947: 10007] = 169
    labels[10007: 10067] = 170
    labels[10067: 10127] = 171
    labels[10127: 10187] = 172
    labels[10187: 10247] = 173
    labels[10247: 10307] = 174
    labels[10307: 10367] = 175
    labels[10367: 10427] = 176
    labels[10427: 10483] = 177
    labels[10483: 10542] = 178
    labels[10542: 10602] = 179
    labels[10602: 10661] = 180
    labels[10661: 10721] = 181
    labels[10721: 10781] = 182
    labels[10781: 10841] = 183
    labels[10841: 10901] = 184
    labels[10901: 10961] = 185
    labels[10961: 11011] = 186
    labels[11011: 11071] = 187
    labels[11071: 11131] = 188
    labels[11131: 11189] = 189
    labels[11189: 11249] = 190
    labels[11249: 11309] = 191
    labels[11309: 11369] = 192
    labels[11369: 11429] = 193
    labels[11429: 11489] = 194
    labels[11489: 11548] = 195
    labels[11548: 11608] = 196
    labels[11608: 11668] = 197
    labels[11668: 11728] = 198
    labels[11728: 11788] = 199


    


    # Labels
    names=['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']



    # Convert class labels to one-hot encoding
    Y=np_utils.to_categorical(labels, 200)
    #Shuffle the dataset
    x,y=shuffle(img_data,Y,random_state=2)
    print(x.shape, y.shape)
    #Split the dataset to train and test
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

    # the data, shuffled and split between train and test sets
    # from keras.datasets import mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # y_train = to_categorical(y_train.astype('float32'))
    # y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('-f')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test), args=args)
