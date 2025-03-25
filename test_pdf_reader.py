import os 
from src.pdf_processor import PDFProcessor


def test_pdF_extract(): 
    pdf_path = 'data/Google Code of Conduct.pdf'

    processor = PDFProcessor(chunk_size=500, overlap=100)
    extract_text = processor.extract_text(pdf_path)

    print("Extract text")
    # print(extract_text) 


    chunks = processor.chunk_text(extract_text)

    print(f"S-au extras {len(chunks)}")

    num_samples = min(3, len(chunks))

    for i in range(num_samples):
        print(f"\nChunk #{i+1} ({len(chunks[i])} characters):")
        print("-" * 40)
        print(chunks[i][:200] + "...")  
        print("-" * 40)

    chunk_lengths = [len(chunk) for chunk in chunks]
    avg_length = sum(chunk_lengths) / len(chunks) if chunks else 0 
    print(avg_length)




if __name__=="__main__":
    test_pdF_extract()