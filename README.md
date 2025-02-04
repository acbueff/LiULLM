# LiULLM

1. set up global paths and machine name and container libarary in:
global_scripts/global_configuration.sh

2. configure path concering the LLM foundary environment to your liking in:
llm-foundry-env/configuration.sh

3. swith into the llm foundary environment:
cd llm-foundry-env

4. builds the llm foundry container
nice bash build_container.sh

5. to prevent overwriting the actively used container, we use a different path to build the container. 
This means we have to mve the container to another location, where we expect the active container to be:
bash move_built_container_to_active.sh

6. Set up the rest of the software stack:
nice bash set_up.sh

9. SSH onto a CPU-focused supercomputer such as JUWELS Cluster for CPU tasks.

10. Preprocess the validation data, tokenixing it and converting it to the binary format used by LLM foundry. 
Here is an example assuming you're on the JUWELS Cluster:
    sbatch jsc/preprocess_data_jwc.sbatch

11. You can also preprocess data in parallel; we will do this for the training data. Here is an example, again assuming you're on the JUWELS Cluster:
    sbatch jsc/preprocess_data_parallel_jwc.sbatch
    - You need to execute both, the parallel and non-parallel preprocessing examples, for the training example to work, because they create different splits 

12. SSH onto the GPU-focused supercomputer such as JUWELS Booster for GPU tasks:
    sbatch jsc/train_jwc.sbatch

13. Run a pretraining task. Here is an example assumeing youre on the JUWELS Booster:
    sbatch jsc/run_training_jwb.sbatch

