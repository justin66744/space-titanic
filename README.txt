Best Kaggle Submission Score: 0.79237

Features used with reasoning:
    CryoSleep:

    Predicts quite well if a passenger was shipped. Passengers in CryoSleep likely had a higher chance of having been shipped.

    Age:

    Survival rates may be influenced by age, as younger travelers might have had a higher chance of being shipped.

    RoomService, Spa, VRDeck, FoodCourt, ShoppingMall:

    These are cost features that explain how much the passenger had paid for facilities. 
    Passengers who were most likely to have paid more may have had a greater chance of getting transported as they were perhaps active or with statuses.

    HomePlanet:

    The home planet of the passenger might also decide on the chances of getting transported,
    considering the reality that different planets might have varying emergency evacuation measures in place.

    Destination:

    The target planet could also have influenced probability of transportation because certain targets might have been safer or more of a priority.

    VIP:

    VIP passengers could have had a greater likelihood of being transported due to their status.




Methods for accuracy:

- Trained across a range of n_neighbors values (from 1 to 50).
in an attempt to determine the KNN algorithm's ideal neighbor count. 


- Cross-validation was used to assess the model's performance on various subsets
  of the training data and make sure it performed well when applied to fresh data.

- Features that were unlikely to have a significant impact on the model's performance,
  such as PassengerId, Name, and Cabin, were dropped.

