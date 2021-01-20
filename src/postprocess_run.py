# import settings
# import const
# import sys
# from src import postprocess as post
#
#
# if __name__ == '__main__':
#     do_3types = True
#     do_3types_weighted = True
#     similarity_type = "distance"     # distance product
#     do_sine = True
#     sine_model = "35.p"
#     # ----------------- Node2Vec on 3 Types ----------------- #
#     if do_3types:
#         post.n2v_save_embeddings_to_mat(embed_path=settings.config[const.SLASHDOT_3TYPE_OUTPUT],
#                                         save_path=settings.config[const.SLASHDOT_3TYPE_EMBED])
#         # create train dataset from embeddings
#         post.generate_dataset_3types(embed_path=settings.config[const.SLASHDOT_3TYPE_EMBED],
#                                      graph_path=settings.config[const.SLASHDOT_3TYPE_TRAIN],
#                                      new2old_path=settings.config[const.SLASHDOT_3TYPE_NEW2OLD],
#                                      old2new_path=settings.config[const.SLASHDOT_3TYPE_OLD2NEW],
#                                      save_path=settings.config[const.SLASHDOT_3TYPE_LINK_TRAIN])
#         # create test dataset from embeddings
#         post.generate_dataset_3types(embed_path=settings.config[const.SLASHDOT_3TYPE_EMBED],
#                                      graph_path=settings.config[const.SLASHDOT_3TYPE_TEST],
#                                      new2old_path=settings.config[const.SLASHDOT_3TYPE_NEW2OLD],
#                                      old2new_path=settings.config[const.SLASHDOT_3TYPE_OLD2NEW],
#                                      save_path=settings.config[const.SLASHDOT_3TYPE_LINK_TEST])
#         print('3types done.')
#     # ----------------- Node2Vec on 3 Types Weighted ----------------- #
#     if do_3types_weighted:
#         post.n2v_save_embeddings_to_mat(embed_path=settings.config[const.SLASHDOT_UNSIGNED_OUTPUT],
#                                         save_path=settings.config[const.SLASHDOT_UNSIGNED_EMBED])
#         # create train dataset from embeddings
#         post.generate_dataset_attention(embed_path=settings.config[const.SLASHDOT_3TYPE_EMBED],
#                                         unsigned_embed_path=settings.config[const.SLASHDOT_UNSIGNED_EMBED],
#                                         relations_path=settings.config[const.SLASHDOT_3TYPE_TRAIN],
#                                         graph_path=settings.config[const.SLASHDOT_3TYPE_TRAIN],
#                                         new2old_path=settings.config[const.SLASHDOT_3TYPE_NEW2OLD],
#                                         old2new_path=settings.config[const.SLASHDOT_3TYPE_OLD2NEW],
#                                         save_path=settings.config[const.SLASHDOT_UNSIGNED_LINK_TRAIN],
#                                         similarity_type=similarity_type)
#         # create test dataset from embeddings
#         # whole (test + train) graph can be used for relations
#         post.generate_dataset_attention(embed_path=settings.config[const.SLASHDOT_3TYPE_EMBED],
#                                         unsigned_embed_path=settings.config[const.SLASHDOT_UNSIGNED_EMBED],
#                                         relations_path=settings.config[const.SLASHDOT_3TYPE],
#                                         graph_path=settings.config[const.SLASHDOT_3TYPE_TEST],
#                                         new2old_path=settings.config[const.SLASHDOT_3TYPE_NEW2OLD],
#                                         old2new_path=settings.config[const.SLASHDOT_3TYPE_OLD2NEW],
#                                         save_path=settings.config[const.SLASHDOT_UNSIGNED_LINK_TEST],
#                                         similarity_type=similarity_type)
#         print('3types weighted done.')
#     # ------------------- SINE ------------------- #
#     if do_sine:
#         post.sine_save_embeddings_to_mat(model_path=settings.config[const.SLASHDOT_EMBED_MODEL] + sine_model,
#                                          save_path=settings.config[const.SLASHDOT_SINE_EMBED])
#         print('embeddings converted to matrix, and saved')
#         # create train dataset from embeddings
#         post.generate_dataset_sine(embed_path=settings.config[const.SLASHDOT_SINE_EMBED],
#                                    graph_path=settings.config[const.SLASHDOT_GRAPH_TRAIN],
#                                    save_path=settings.config[const.SLASHDOT_LINK_TRAIN])
#         print('train dataset generated for link prediction')
#         # create test dataset from embeddings
#         post.generate_dataset_sine(embed_path=settings.config[const.SLASHDOT_SINE_EMBED],
#                                    graph_path=settings.config[const.SLASHDOT_GRAPH_TEST],
#                                    save_path=settings.config[const.SLASHDOT_LINK_TEST])
#         print('test dataset generated for link prediction')
#     sys.exit(0)
