
import numpy as np
import tensorflow as tf
#from model_ops_2 import cell1D,cell2D_res,CONV2D,BatchNorm
#import signed_dist_functions as SDF


def normalize_vector(vector):
  return vector / tf.sqrt(tf.reduce_sum(tf.square(vector), axis=0))

def vector_fill(shape, vector):
  return tf.stack([
    tf.fill(shape, vector[0]),
    tf.fill(shape, vector[1]),
    tf.fill(shape, vector[2]),])



    
    

class Raycast(object):
    def __init__(self, batch_size, resolution=(200,200), sky_color = [70., 130., 180.]):
        self.resolution = resolution
        self.cameras = []
        self.sky_color = sky_color
        self.batch_size = batch_size




    def add_lighting(self,position=[0., 1., 1.],color=[255., 255., 255.],ambient_color = [119., 139., 165.]):
        
        light_position = normalize_vector(np.array(position,dtype=np.float32))        
        self.light = {"position": light_position,
                 "color": np.array(color,dtype=np.float32)}
        self.ambient_color = ambient_color
        
        

        
    def add_camera(self,camera_position=[-2., 0., 0.],lookAt=(0,0,0),focal_length=1,name='camera_1'):
        with tf.name_scope(name):
            with tf.variable_scope(name+'_ray',reuse=tf.AUTO_REUSE):
                Cam = Camera(self.resolution,camera_position,lookAt,focal_length,self.batch_size)
                self.cameras.append(Cam)
 

    def eval_func(self,model_fn,args,step_size, epsilon, temp, ambient_weight):
        renders = []
        clouds  = []
        distances = []
        ray_steps=[]
        camera_resets = []
        backgrounds = []
        images = []
        raw_images = []
        normals = []
        incidence_angles=[]
        for i in range(len(self.cameras)):
            ray_step,cloud,dist = self.cameras[i].evaluate(model_fn,args,step_size,epsilon,temp)
            render,background,image,raw_image,normal,incidence_angle       = self.cameras[i].compose_image(self.light,self.ambient_color,self.sky_color,epsilon,ambient_weight=ambient_weight)
            camera_resets.append(self.cameras[i].reset_camera())
            ray_steps.append(ray_step)
            renders.append(render)
            clouds.append(cloud)
            distances.append(dist)
            backgrounds.append(background)
            images.append(image)
            raw_images.append(raw_image)
            normals.append(normal)
            incidence_angles.append(incidence_angle)
        return ray_steps,camera_resets,{'renders':renders,
                                        'clouds':clouds,
                                        'distances':distances,
                                        'backgrounds':backgrounds,
                                        'images':images,
                                        'raw_images':raw_images,
                                        'normals':normals,
                                        'incidence_angles':incidence_angles}
                                


    def trace(self,session,ray_steps,reset_opp,num_steps=50):
        session.run(reset_opp)
        step = 0
        while step < num_steps:
            session.run(ray_steps)
            step += 1
#        print('  sphere tracing')













class Camera(object):
    def __init__(self,resolution,camera_position=[-2., 0., 0.],lookAt=(0.,0.,0.),focal_length=1,batch_size=1):
        self.focal_length=focal_length
        # Find the center of the image plane
        self.batch_size = batch_size
        if camera_position=='random':
            location = np.random.uniform(size=(3,),low=-1.,high=1.).astype(np.float32)
            location = location/np.sqrt(np.sum(np.square(location)))
            distance = np.random.uniform(size=(1,),low=0.99,high=1.01).astype(np.float32)
            camera_position = location*distance
            
        self.camera_position = tf.constant(camera_position)
        self.camera = self.camera_position - np.array(lookAt)
        self.camera_direction = normalize_vector(self.camera)
        self.eye = self.camera + self.focal_length * self.camera_direction  
        self.resolution = resolution
        self.resolutions = list(map(lambda x: x*1j, self.resolution))
        self.aspect_ratio = 1.*resolution[0]/resolution[1]
        self.min_bounds, self.max_bounds = (-self.aspect_ratio, -1), (self.aspect_ratio, 1)        
        self.image_plane_coords = np.mgrid[self.min_bounds[0]:self.max_bounds[0]:self.resolutions[0],self.min_bounds[1]:self.max_bounds[1]:self.resolutions[1]]
        
        self.add_sensor()
        self.parametrize_rays()



    def add_sensor(self):
        # Coerce into correct shape
        self.image_plane_center = vector_fill(self.resolution, self.camera_position)
        # Convert u,v parameters to x,y,z coordinates for the image plane
        self.u_unit = [1., 0., 0.]
        self.v_unit = tf.cross(self.camera_direction, self.u_unit)
        self.image_plane = self.image_plane_center + self.image_plane_coords[0] * vector_fill(self.resolution, self.u_unit) + self.image_plane_coords[1] * vector_fill(self.resolution, self.v_unit)
        # Populate the image plane with initial unit ray vectors
        self.initial_vectors  = self.image_plane - vector_fill(self.resolution, self.eye)
        self.ray_vectors      = normalize_vector(self.initial_vectors)  
            
    def parametrize_rays(self):
        self.t = tf.get_variable(name="ScalingFactor", shape=(self.resolution[0],self.resolution[1]), initializer=tf.zeros_initializer(),trainable=True)
        space = (self.ray_vectors * self.t) + self.image_plane
        # Name TF ops for better graph visualization
        self.x = tf.squeeze(tf.slice(space, [0,0,0], [1,-1,-1]), squeeze_dims=[0], name="X-Coordinates")
        self.y = tf.squeeze(tf.slice(space, [1,0,0], [1,-1,-1]), squeeze_dims=[0], name="Y-Coordinates")
        self.z = tf.squeeze(tf.slice(space, [2,0,0], [1,-1,-1]), squeeze_dims=[0], name="Z-Coordinates")
        self.coordinates     = tf.expand_dims(tf.stack((self.x,self.y,self.z),axis=-1),axis=0)
#        self.coordinates_min = tf.get_variable(name="MinimumCrossing", initializer=self.coordinates,trainable=False)
        
    def reset_camera(self):
#        camera_reset = [self.t.assign(tf.zeros(self.t.shape)),self.coordinates_min.assign(self.coordinates)]
        camera_reset = self.t.assign(tf.zeros(self.t.shape))
        space = (self.ray_vectors * self.t) + self.image_plane
        # Name TF ops for better graph visualization
        self.x = tf.squeeze(tf.slice(space, [0,0,0], [1,-1,-1]), squeeze_dims=[0], name="X-Coordinates")
        self.y = tf.squeeze(tf.slice(space, [1,0,0], [1,-1,-1]), squeeze_dims=[0], name="Y-Coordinates")
        self.z = tf.squeeze(tf.slice(space, [2,0,0], [1,-1,-1]), squeeze_dims=[0], name="Z-Coordinates")
        self.coordinates = tf.expand_dims(tf.stack((self.x,self.y,self.z),axis=-1),axis=0)
        return camera_reset
        
        
    def evaluate(self,eval_fn,args, step_size = None, epsilon = 0.0001,temp = 1. ):
        evaluated_function = eval_fn(self.coordinates,args)
        # Ray tracing step size 
        self.evaluated_function = tf.squeeze(evaluated_function,axis=0)
        # Iteration operation
        self.distance      = tf.abs(self.evaluated_function)
        if step_size==None:
            self.distance_step = self.t + (tf.sign(self.evaluated_function) * tf.maximum(self.distance, epsilon))
        else:
            self.distance_step = self.t + (tf.sign(self.evaluated_function) * tf.maximum(step_size, epsilon))
        self.ray_step      = self.t.assign(self.distance_step)
        
        # calculate normals
        gradient = tf.stack(tf.gradients(self.evaluated_function, [self.x,self.y,self.z]))
        self.normal_vector = normalize_vector(gradient)
        self.point_cloud = self.coordinates
        return self.ray_step,self.point_cloud,self.distance




    
    def compose_image(self,light,ambient_color ,sky_color = [70., 80., 180.],epsilon= 0.0001,ambient_weight = 0.1):
        normals = self.normal_vector
#        incidence = normals - vector_fill(self.resolution, light["position"])
#        normalized_incidence = normalize_vector(incidence)
#        incidence_angle = tf.reduce_sum(normalized_incidence * normals, reduction_indices=0)        

#        incidence = vector_fill(self.resolution, light["position"])
        incidence            = vector_fill(self.resolution, light["position"]) - tf.stack((self.x,self.y,self.z),axis=0)
        normalized_incidence = normalize_vector(incidence)
        incidence_angle      = (tf.reduce_sum(normalized_incidence * normals, reduction_indices=0)+1.)/2.
        # Split the color into three channels
        light_intensity = vector_fill(self.resolution, light['color']) * incidence_angle
        # Add ambient light
        with_ambient   = light_intensity * (1.-ambient_weight) + vector_fill(self.resolution, ambient_color) * ambient_weight
        # Mask out pixels not on the surface
#        bitmask        = tf.less_equal(self.distance, epsilon)
        bitmask        = tf.exp(-self.distance)
        
        lighted        = with_ambient
        masked         = lighted * tf.to_float(bitmask)
        background     = vector_fill(self.resolution, sky_color) * tf.to_float(1.-bitmask)
        raw_image_data = tf.transpose(masked + background)


        # Render 
        image  = tf.cast(raw_image_data, tf.uint8)
        render = tf.image.encode_jpeg(image) 
        return render,background,image,raw_image_data,normals*tf.to_float(bitmask),incidence_angle
        



#input_ = tf.constant(10.)
#@tf.custom_gradient
#def clip_grad_layer(x):
#    smooth = x**2
#    def grad(dy):
#        return dy*tf.gradients(smooth,x)[0]
#    return tf.identity(x), grad
#
#output_clip = clip_grad_layer(input_)
#grad_clip = tf.gradients(output_clip, input_)
## output without gradient clipping in the backwards pass for comparison:
#output = tf.identity(input_)
#grad = tf.gradients(output, input_)
#
#with tf.Session() as sess:
#  sess.run(tf.global_variables_initializer())
#  print("input:", sess.run(input_))
#  print("output:", sess.run(output_clip))  
#  print("with clipping:", sess.run(grad_clip)[0])
#  print("without clipping:", sess.run(grad)[0])
