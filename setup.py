import setuptools

setuptools.setup(
    name='jacobian',
    version='1.0',
    author='Felix Kempf',
    author_email='felix.kempf@kcl.ac.uk',
    description='Jacobian Regularisation of Input gradients',
    url='https://github.com/fkempf92/JacobianRegularisation',
    project_urls = {
        "Bug Tracker": "https://github.com/fkempf92/JacobianRegularisation/issues"
    },
    license='MIT',
    packages=['jacobian'],
    install_requires=['torch >= 1.5.1',
                      'skorch >= 0.11.0'],
)
