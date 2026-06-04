def encode_patch_tokens(batch , visual, image):
    B = batch
    #image shape : (B, 3, 240, 240)
    x = visual.conv1(image)
    #visual = vit-b/16, embedding = 768
    #x.shape = (B, 768, 15, 15), (16, 16) patch들이 15, 15 형태로 존재.
    patch_tokens = x.reshape(B, 768, 225).permute(0,2,1)
    #flatten을 하는 이유는 transformer는 sequence를 입력으로 받기때문에.
    #그리고 225는 각 grid에 들어있던 patch들이 펼처진것.
    window_patch_tokens = patch_tokens[:, mask]
    #pytorch에서 위 코드는 patch_tokens[: , mask, :]의 축양형이다.
