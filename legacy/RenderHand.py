
import tensorflow as tf
import numpy as np
from src.utilities import raytrace as RAY


#        import sys
#        sys.path.append('/Users/gidilittwin/Dropbox/Thesis/ModelNet/')
#        import RenderHand as RH
#        RH.render_dist_functions(TF.dist_functions,hand_model,kl_obj,config,feed_dict)
                
        


BATCH_SIZE = 1
def render_dist_functions(function_,hand_model,kl_obj,config,feed_dict):


    #%% Initiate renderers 
    with tf.variable_scope('display_tracer',reuse=tf.AUTO_REUSE):
        Ray_render = RAY.Raycast(BATCH_SIZE,resolution=(500,500))
        Ray_render.add_lighting(position=[0., 1., 1.],color=[255., 255., 255.],ambient_color = [244., 83., 66.])
        Ray_render.add_camera(camera_position=[0.8, 0.8, 0.8], lookAt=(0,0,0),focal_length=1,name='camera_1')
    
    
    
        
    def function_wrapper(coordinates,args_):
        with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
    #        evaluated_function = SF.deep_sdf2(coordinates,args_[0],args_[1])
            shape_ = coordinates.get_shape().as_list()
            coordinates = tf.reshape(coordinates,(1,-1,3))
            evaluated_function = args_[0](args_[1],coordinates,args_[2],args_[3])
            evaluated_function = tf.reshape(evaluated_function,(1,shape_[1],shape_[2]))
            
            return evaluated_function
    args_=[function_,hand_model,kl_obj,config]        
    
    
    
    
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    step = 0
    
    
    ray_steps,reset_opp,evals = Ray_render.eval_func(model_fn = function_wrapper, args=args_,step_size=0.004, epsilon = 0.0001, temp=1.)
    session.run(reset_opp)
    step = 0
    while step < 50:
        session.run(ray_steps,feed_dict=feed_dict)
        step += 1
        print(step)
    evals_ = session.run(evals,feed_dict=feed_dict) # <= returns jpeg data you can write to disk
    
    for i in range(len(evals_['renders'])):    
        with open('/Users/gidilittwin/Desktop/Renders/image'+str(step)+'_'+str(i)+'.jpg', 'w') as fd:
            fd.write(evals_['renders'][i])
                
    
    
