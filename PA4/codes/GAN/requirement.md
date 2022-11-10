1.  Train a GAN with default hyperparameters. Increase either latent_dim or hidden_dim from 16 to 100. How about changing both to 100? (**You need to run at least four experiments in this homework.**)

2. The code we provide will plot a number of graphs (e.g. loss curves) during training on the
   Tensorboard. Show them in your report. (5 curves for a GAN. Make sure that TAs can see each curve clearly.)
3. The code we provide will also paint some images on the Tensorboard. Show the model-generated images at the last record step. (One group of images for a GAN. Make sure that TAs can see each graph clearly.)
4. Report FID scores of the four trained GANs mentioned above. (The best GAN's FID should be lower than 60.)
5. Discuss how latent_dim and hidden_dim influence the performance of a GAN.
6. Has your GAN converged to the Nash equilibrium? Give your explanation with your training curves.
7. Choose the best GAN, add some codes and apply linear interpolation in the latent space. Show the interpolated images (10 images for a pair $z_1$ and $z_2$, at least 5 pairs) and describe what you see (the connections between interpolated images and the generation performance).
8. Choose one of your GANs and investigate whether it suffers from the mode collapse problem. You can manually do the labelling or use the MNIST classifier in HW1.
9.  Implement a GAN with an MLP-based generator and discriminator. Report your FID score and show no less than 10 generated images. Compare its performance with the CNN-based GAN and try to explain what makes the differences.

Since the training time may be long, you are not required to tune the hyper-parameters. The default hyper-parameters are enough to produce reasonable results (if you have implemented the model correctly, especially for the loss, which should be averaged over the batch).