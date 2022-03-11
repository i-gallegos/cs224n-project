from transformers import AutoTokenizer, AutoModel, utils, BartModel, BartConfig
from bertviz import model_view, show
#from bertviz.head_view_bert import show
from IPython import display

utils.logging.set_verbosity_error()  # Remove line to see warnings

def call_html():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

# Initialize tokenizer and model. Be sure to set output_attentions=True.
# Load BART fine-tuned for summarization on CNN/Daily Mail dataset
full_checkpoints = ['tldr_lr3e-05_seed161_full_datasetTrue/checkpoint-4', "tosdr_lr3e-05_seed161_full_datasetTrue/checkpoint-8", "small_billsum_lr3e-05_seed224_full_datasetTrue/checkpoint-52"]
within_checkpoints = ['tldr_lr3e-05_seed161_full_datasetFalse/checkpoint-4', 'tosdr_lr3e-05_seed161_full_datasetFalse/checkpoint-8', 'small_billsum_lr3e-05_seed161_full_datasetFalse/checkpoint-44']
checkpoint = full_checkpoints[0]
config = checkpoint+"/config.json"
#model_name = f"facebook/bart-large-cnn"
model_path = f"bart-large-cnn-finetuned/{checkpoint}"
#model = BartModel(config)
#configuration = model.config
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, output_attentions=True, config=config)

sentenceA = "The House Budget Committee voted Saturday to pass a $3.5 trillion spending bill"
sentenceB = "The House Budget Committee passed a spending bill."
# get encoded input vectors
encoder_input_ids = tokenizer(sentenceA, return_tensors="pt", add_special_tokens=True).input_ids

# create ids of encoded input vectors
decoder_input_ids = tokenizer(sentenceB, return_tensors="pt", add_special_tokens=True).input_ids

outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)

encoder_text = tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
decoder_text = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

model_view(
    encoder_attention=outputs.encoder_attentions,
    decoder_attention=outputs.decoder_attentions,
    cross_attention=outputs.cross_attentions,
    encoder_tokens= encoder_text,
    decoder_tokens=decoder_text
)

call_html()
show(model, tokenizer, sentenceA, sentenceB)