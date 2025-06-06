import torch, transformers, tqdm

tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

class ICVModel(torch.nn.Module): 
    
    def __init__(self) -> None:
        super().__init__()

        self.model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        self.num_h = len(self.model.model.layers)
        
        for param in self.model.parameters(): 
            param.requires_grad = False

        self.icvs = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(1, 1, self.model.config.hidden_size))
            for i in range(self.num_h)
        )

        for i in range(self.num_h):
            self.model.model.layers[i].register_forward_hook(self.create_hook(i))
    
    def create_hook(self, i: int): 
        def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
            return output + self.icvs[i]
        return hook_fn

    def forward(self, x): 
        return self.model(x)

    
model = ICVModel()

