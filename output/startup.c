
/* Minimal bare-metal startup for Hazard3 RISC-V */
extern void model_inference(const float* input, float* output);

/* Minimal soft-float math stubs if needed */
#ifndef __riscv_float_abi_soft
/* Use compiler builtins */
#endif

/* Simple entry point */
void _start(void) {
    /* Placeholder: in real deployment, load inputs from memory-mapped I/O */
    static float input[1] = {0.0f};
    static float output[1] = {0.0f};

    model_inference(input, output);

    /* Halt: write to test-finish register or infinite loop */
    while(1) {
        __asm__ volatile ("wfi");
    }
}
