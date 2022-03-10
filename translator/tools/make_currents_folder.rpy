def make_currents_folder():
    make_folder('currents')
    set_current_directory('currents')
    for x in 'a2b b2a'.split():
        for y in 'test train'.split():
            name='gen_%s_%s_current.png'%(x,y)
            make_symlink('../%s'%name,name)