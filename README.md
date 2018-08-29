# **Exploring State Transition Uncertainty in Deep Reinforcement Learning**
## **Introduction**
- We apply **transition uncertainty critic** (**TUC**) to coworking with agent in four experiments

- Experiments consist of **cart pole**、**grid world**、**MNIST tracking** and **hierarchical object detection**


- Our agent has two parts 
  - **actor** : select action according to state 
  - **TUC** : generate intrinsic signals to make actor explore efficiently

- Block digram of TUC and actor
  <center>
  <img src="https://i.imgur.com/teBa132.png" width="40%" height="40%" />
  </center>

- Overview of TUC and actor
  <center>
  <img src="https://i.imgur.com/kUvDmzX.png" width="40%" height="40%" />
  </center>
  
  - KL divergence stands for the uncertainty of state transition ( exploration )
  - Penalty term makes actor to perform better ( exploitation )
  - For example, consider a **Markov decision process** which consists of 2 states and 2 actions  
    <center>
    <img src="https://i.imgur.com/nlygbFx.png" width="30%" height="30%" />
    </center>
  - **Action-gating mechanism** is applied to model-based module to learn the **latent variables** corresponding to **transition** of **different action**
  
  

## **Settings**

- **Hardware** 
  - CPU : Intel Xeon E7-2620 @2.00 GHz
  - RAM : 32 GB DDR4-2400
  - GPU : Tesla P100
- **Deep Learning API** 
  - Tensorflow 1.3
  - Pytorch 0.4

## **Results**
- **Cart Pole**

  <img src="https://i.imgur.com/ZxJHcJX.png" width="50%" height="50%" />
  
  - Total extrinsic reward curve
    - Extrinsic reward 1 is given until termination 
    
    <img src="https://i.imgur.com/P2icY98.png" width="40%" height="40%" />

  - Latent features
  
    <img src="https://i.imgur.com/KP9kETp.png" width="30%" height="30%" /> <img src="https://i.imgur.com/P7TCiAh.png" width="30%" height="30%" />
    
    <img src="https://i.imgur.com/XhFqS6Z.png" width="30%" height="30%" /> <img src="https://i.imgur.com/sKOzII1.png" width="30%" height="30%" />
    

    
  

   
    
- **Grid World**

  <img src="https://i.imgur.com/Auz6ClT.png" width="30%" height="30%" />
  
  - Total extrinsic reward curve
    - Extrinsic reward 5 is given when agent ( red square ) catches the taget ( blue circle )
    - Extrinsic reward -1 is given when agent ( red square ) catches the obstacle ( green triangle )
  
    <img src="https://i.imgur.com/pWOKSGd.png" width="40%" height="40%" />
    
  - Latent features 
  
    <img src="https://i.imgur.com/xQbfNgB.png" width="30%" height="30%" /> <img src="https://i.imgur.com/HKFsV5d.png![]" width="30%" height="30%" />
    
    <img src="https://i.imgur.com/BtY6bGl.png![]" width="30%" height="30%" /> <img src="https://i.imgur.com/dnhv2td.png" width="30%" height="30%" />


- **Tracking MNIST**
  - Network and interation
    - Extrinsic reward 1 is given when IoU ( Intersection-over-Union ) > 0.7
    - Extrinsic reward -0.1 is given when IoU ( Intersection-over-Union ) < 0.7
    <center>
    <img src="https://i.imgur.com/bw7duKi.png" width="50%" height="50%" />
    </center>
    
  - Success and precision
  
    <img src="https://i.imgur.com/IXJoBpt.png" width="40%" height="40%" />  <img src="https://i.imgur.com/ijfQURe.png" width="40%" height="40%" />

  - Tracking trajectories
  
    <img src="https://i.imgur.com/h9mk6Jh.png" width="50%" height="50%" />

- **Hierarchical Object Detection**

  - Network and interation
    - Before termination
      - Extrinsic reward 1 is given when IoU ( Intersection-over-Union ) > 0.5
      - Extrinsic reward -1 is given when IoU ( Intersection-over-Union ) < 0.5
    - At termination
      - Extrinsic reward 6 is given when IoU ( Intersection-over-Union ) > 0.5
      - Extrinsic reward -6 is given when IoU ( Intersection-over-Union ) < 0.5
    <center>
    <img src="https://i.imgur.com/musr3V3.png" width="50%" height="50%" />
    </center>

  - Precision-Recall curve
  
    <img src="https://i.imgur.com/mqqRU4k.png" width="40%" height="40%" />

  - Hierarchical detection procedure
  
    <img src="https://i.imgur.com/oFpcr2U.jpg" width="50%" height="50%" />


  

