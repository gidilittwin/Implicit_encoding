{
    "grid_size"       : 36,
    "checkpoint_every": 50000,
    "plot_every"      : 1000,
    "test_every"      : 10000,
    "learning_rate"   : 0.00001,
    "levelset"        : 0.0,
    "batch_size"      : 8,
    "num_samples"     : 10000,
    "global_points"   : 10000,
    "path"            : "/media/gidi/SSD/Thesis/Data/ShapeNetRendering/",
    "train_file"      : "/media/gidi/SSD/Thesis/Data/ShapeNetRendering/train_list.txt",
    "test_file"       : "/media/gidi/SSD/Thesis/Data/ShapeNetRendering/test_list.txt",
    "categories"      : ["02691156","02828884","02933112","02958343","03001627","03211117","03636649","03691459","04090263","04256520","04379243","04401088","04530566"],
    "checkpoint_path" : "/media/gidi/SSD/Thesis/Data/Checkpoints/exp31(benchmark=57.4)/",
    "finetune"        : true,
    "saved_model_path": "/media/gidi/SSD/Thesis/Data/Checkpoints/exp31(benchmark=57.4)/-196069",
    
    "theta"           : [{"w":32,"in":3}, {"w":32,"in":32}, {"w":32,"in":32}, {"w":1 ,"in":32}],
    "decoder"         : [{"size":512,"act":true, "batch_norm":false},
                         {"size":512,"act":false,"batch_norm":false}],
    
    "encoder"         : {"base_size":32, "BN":true, "input":{"k":5,"stride":1},
                            "residuals":[{"k":3, "s_in":1, "s_out":1, "stride":2},
                                         {"k":3, "s_in":1, "s_out":1, "stride":1},
                                         {"k":3, "s_in":1, "s_out":1, "stride":1},
                                         
                                         {"k":3, "s_in":1, "s_out":2, "stride":2},
                                         {"k":3, "s_in":2, "s_out":2, "stride":1},
                                         {"k":3, "s_in":2, "s_out":2, "stride":1},
                                         {"k":3, "s_in":2, "s_out":2, "stride":1},
                                         
                                         {"k":3, "s_in":2, "s_out":4, "stride":2},
                                         {"k":3, "s_in":4, "s_out":4, "stride":1},
                                         {"k":3, "s_in":4, "s_out":4, "stride":1},
                                         {"k":3, "s_in":4, "s_out":4, "stride":1},                                        
                                         {"k":3, "s_in":4, "s_out":4, "stride":1},                                        
                                         {"k":3, "s_in":4, "s_out":4, "stride":1}, 
                                         
                                         {"k":3, "s_in":4, "s_out":8, "stride":2},
                                         {"k":3, "s_in":8, "s_out":8, "stride":1},
                                         {"k":3, "s_in":8, "s_out":8, "stride":1},
                                         
                                         {"k":3, "s_in":8, "s_out":16, "stride":2},
                                         {"k":3, "s_in":16, "s_out":16, "stride":1},
                                         {"k":3, "s_in":16, "s_out":16, "stride":1} ]}

}

