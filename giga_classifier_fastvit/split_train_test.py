import os

input_folder = '/mnt/data/giga_classifier/crops'
output_folder = '/mnt/data/giga_classifier/split'
train_split = 0.8

# create output path
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# list all images in input path
images = os.listdir(os.path.join(input_folder, '0'))
# split images
train_images = images[:int(len(images) * train_split)]
test_images = images[int(len(images) * train_split):]

# create output path
if not os.path.exists(os.path.join(output_folder, 'train', '0')):
    os.makedirs(os.path.join(output_folder, 'train', '0'))
    os.makedirs(os.path.join(output_folder, 'test', '0'))

    os.makedirs(os.path.join(output_folder, 'train', '1'))
    os.makedirs(os.path.join(output_folder, 'test', '1'))

# move images to output path
for image in train_images:
    os.rename(os.path.join(input_folder, '0', image), os.path.join(output_folder, 'train', '0', image))
for image in test_images:
    os.rename(os.path.join(input_folder, '0', image), os.path.join(output_folder, 'test', '0', image))


# list all images in input path
images = os.listdir(os.path.join(input_folder, '1'))

# split images
train_images = images[:int(len(images) * train_split)]
test_images = images[int(len(images) * train_split):]

# move images to output path
for image in train_images:
    os.rename(os.path.join(input_folder, '1', image), os.path.join(output_folder, 'train', '1', image))

for image in test_images:
    os.rename(os.path.join(input_folder, '1', image), os.path.join(output_folder, 'test', '1', image))

print('Done')
    