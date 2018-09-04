# **Exploring State Transition Uncertainty in Deep Reinforcement Learning**
## **Introduction**
- We apply **transition uncertainty critic** (**TUC**) to coworking with agent in four experiments

- Experiments consist of **cart pole**、**grid world**、**MNIST tracking** and **hierarchical object detection**


- Our agent has two parts 
  - **actor** : select action according to state 
  - **TUC** : generate intrinsic signals to make actor explore efficiently

- Block digram of TUC and actor
  <center>
  <img src="figures/Total_Framework_Simple.png" width="40%" height="40%" />
  </center>

- Overview of TUC and actor
  <center>
  <img src="figures/Total_Framework.png" width="40%" height="40%" />
  </center>
  
  - KL divergence stands for the uncertainty of state transition ( exploration )
  - Penalty term makes actor to perform better ( exploitation )
  - For example, consider a **Markov decision process** which consists of 2 states and 2 actions  
    <center>
    <img src="figures/MDP_2S2A_2MC.png" width="30%" height="30%" />
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

  <img src="figures/Cart_pole.png" width="50%" height="50%" />
  
  - Total extrinsic reward curve
    - Extrinsic reward 1 is given until termination 
    
    <img src="figures/Cart_Pole_Total_Reward.png" width="40%" height="40%" />

  - Latent features
  
    <img src="figures/Cart_pole_1_With_Action_Gating.png" width="30%" height="30%" /> <img src="figures/Cart_pole_1_Without_Action_Gating.png" width="30%" height="30%" />
    
    <img src="figures/Cart_pole_2_With_Action_Gating.png" width="30%" height="30%" /> <img src="figures/Cart_pole_2_Without_Action_Gating.png" width="30%" height="30%" />
    

    
  

   
    
- **Grid World**

  <img src="figures/Grid_world.png" width="30%" height="30%" />
  
  - Total extrinsic reward curve
    - Extrinsic reward 5 is given when agent ( red square ) catches the taget ( blue circle )
    - Extrinsic reward -1 is given when agent ( red square ) catches the obstacle ( green triangle )
  
    <img src="figures/Grid_World_Total_Reward.png" width="40%" height="40%" />
    
  - Latent features 
  
    <img src="figures/Grid_world_1_With_Action_Gating.png" width="30%" height="30%" /> <img src="figures/Grid_world_1_Without_Action_Gating.png![]" width="30%" height="30%" />
    
    <img src="figures/Grid_world_2_With_Action_Gating.png![]" width="30%" height="30%" /> <img src="figures/Grid_world_2_Without_Action_Gating.png" width="30%" height="30%" />


- **Tracking MNIST**
  - Network and interation
    - Extrinsic reward 1 is given when IoU ( Intersection-over-Union ) > 0.7
    - Extrinsic reward -0.1 is given when IoU ( Intersection-over-Union ) < 0.7
    <center>
    <img src="figures/CNN_GRU.png" width="50%" height="50%" />
    </center>
    
  - Success and precision
  
    <img src="figures/MNIST_tracking_3_Total_OPE_success.png" width="40%" height="40%" />  <img src="figures/MNIST_tracking_3_Total_OPE_precision.png" width="40%" height="40%" />

  - Tracking trajectories
  
    <img src="figures/MNIST_Tracking_Screen_Shots.png" width="50%" height="50%" />

- **Hierarchical Object Detection**

  - Network and interation
    - Before termination
      - Extrinsic reward 1 is given when IoU ( Intersection-over-Union ) > 0.5
      - Extrinsic reward -1 is given when IoU ( Intersection-over-Union ) < 0.5
    - At termination
      - Extrinsic reward 6 is given when IoU ( Intersection-over-Union ) > 0.5
      - Extrinsic reward -6 is given when IoU ( Intersection-over-Union ) < 0.5
    <center>
    <img src="figures/Object_Detection_Architecture.png" width="50%" height="50%" />
    </center>

  - Precision-Recall curve
  
    <img src="figures/Precision_Recall_Curve.png" width="40%" height="40%" />

  - Hierarchical detection procedure
  
    <img src="figures/detection_results.png" width="50%" height="50%" />


  

