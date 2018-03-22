##Script to fit initial DNN models.
library(keras)

##Downloads data from s3.
if(!file.exists("/home/rstudio-user/Pheno_Relevance_DNN/data/Flickr100.tar.gz")){
  s3_cp_cmd <- "aws s3 cp s3://sdmdata/sample_photos/Flickr100.tar.gz /home/rstudio-user/Pheno_Relevance_DNN/data/Flickr100.tar.gz"
  system(s3_cp_cmd,wait=TRUE)
  untar_cmd <- "tar -xf ./data/Flickr100.tar.gz -C ./data/"
  system(untar_cmd,wait=TRUE)
}

##
image_dir <- "/home/rstudio-user/Pheno_Relevance_DNN/data/Volumes/HerbariumOffice/Flickr_sample100_scratch/"

##Simple model trained from scratch with data augmentation and dropout.

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(300, 300, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate=0.25) %>%
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

##Preparing data.
train_dir <- paste(image_dir,"training/",sep="")
val_dir <- paste(image_dir,"testing/",sep="")

train_datagen <- image_data_generator(rescale = 1/255,
                                      rotation_range= 40,
                                      width_shift_range = 0.2,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = TRUE,
                                      fill_mode="reflect")
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  # This is the target directory
  train_dir,
  # This is the data generator
  train_datagen,
  # All images will be resized to 300 x 300
  target_size = c(300, 300),
  batch_size = 50,
  classes=c('flower', 'nonflower'),
  # Since we use binary_crossentropy loss, we need binary labels
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  val_dir,
  train_datagen,
  target_size = c(300, 300),
  batch_size = 50,
  classes=c('flower', 'nonflower'),
  class_mode = "binary"
)

fnames <- list.files(paste(train_dir,"/relevant",sep=""),full.names=TRUE)
img_path <- fnames[[3]]
img <- image_load(img_path, target_size=c(300,300))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1,300,300,3))

augmentation_generator <- flow_images_from_data(
  img_array,
  generator=train_datagen,
  batch_size=1
)
par(mfrow=c(2,2),pty="s",mar=c(1,0,1,0))
for (i in 1:4){
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("relevance_small_2.h5")

##Model based on vgg16 using weights pre-trained on imagenet.

conv_base <- application_vgg16(
  weights="imagenet",
  include_top=FALSE,
  input_shape=c(300,300,3)
)

freeze_weights(conv_base)

model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

##Unfreeze weights to fine-tune final layers.

unfreeze_weights(conv_base, from = "block3_conv1")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

save_model_hdf5(model, "flowers_small_4.h5")
