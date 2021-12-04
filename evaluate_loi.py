def evaluation(model, weight_path, data_loader, labels):
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    dev_pred , dev_true = [], []
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs =  model.forward_custom(input_ids=b_input_ids, attention_mask=b_input_mask, 
                                       head_mask=None, labels=b_labels)
        # Move logits and labels to CPU

        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        dev_pred.extend([list(p) for p in np.argmax(logits, axis=2)])
        dev_true.extend(label_ids)

    #############################################
    dev_pred_tags = [tag_values[p_i] for p, l in zip(dev_pred, dev_true)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    dev_true_tags = [tag_values[l_i] for l in dev_true
                                  for l_i in l if tag_values[l_i] != "PAD"]
    report = classification_report(y_true=dev_true_tags, y_pred=dev_pred_tags, labels = labels, digits = 4)
    return report