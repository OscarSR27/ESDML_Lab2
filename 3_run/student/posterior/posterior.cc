#include <stdint.h>
#include <stdio.h>

#include "posterior.h"

#include "esp_log.h"
#include "esp_timer.h"
 //Testing
#include <inttypes.h>

/**
 * @brief Default constructor for posterior handler
 *
 * @param history_length Number of past model outputs du consider.
 * @param trigger_threshold_single Threshold value between 0 and 255 for moving average.
 * @param suppression_ms For how many my a new detection should be ignored.
 * @param category_count Number of used labels.

 */
PosteriorHandler::PosteriorHandler(uint32_t history_length, uint8_t trigger_threshold_single,
                                   uint32_t suppression_ms, uint32_t category_count)
    : posterior_history_length_(history_length),
      posterior_trigger_threshold_(trigger_threshold_single * history_length),
      posterior_suppression_ms_(suppression_ms),
      posterior_category_count_(category_count),
      last_detection_time_(0), //Initialize last_detection_time_ to 0
      posterior_history_(nullptr),  // Initialize to nullptr for safe handling.
      moving_average_(nullptr) {// Initialize to nullptr for safe handling. {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  /*
   * Hints:
   * - data structured defined in (posterior.h) have to be initialized here
   * - Normally an embedded developer wouln;t use malloc() to dynamically allocate arrays etc.
   * - However to enable unit testing the history_length as well as the category_count are
   *   not constant and therefore allocation has to be done i.e. using malloc.
   * - While you are allowed to use C++ data structures, it is completely fine if you just
   *   use plain C arrays/pointers/...
   */

  // Allocate memory for the posterior history.
  // 1. Create an array of pointers size = number of classes (posterior_category_count_)
  // 2. For each class create anoher array of pointers size = posterior_history_length_
  // So, this is a bidirectional array rows = posterior_category_count_ and columns = posterior_history_length_ using dynamic memory. What we store here are addresses to
  //memory locations that contains the output of the model. Doing this with pointers allow us to manipulate the size of the array during execution time

  posterior_history_ = new uint8_t*[posterior_category_count_];//First dim, assign dynamic memory to create an array of pointers, the size of this array is posterior_category_count_ and type is uint8_t
  for (uint32_t i = 0; i < posterior_category_count_; ++i) 
  {
    posterior_history_[i] = new uint8_t[posterior_history_length_]();//Second dim, assign dynamic memory to create an array of arrays of pointers, the size of this array 
                                                                     //is posterior_category_count_ x posterior_history_length_ and type is uint8_t. Initialize to zero.
  }

  // Allocate memory for the moving average.
  moving_average_ = new uint32_t[posterior_category_count_]();  // Create an array of pointers with size = posterior_category_count_. Initialize to zero.
                                                                // This array will have the average of N model outputs for each category, where N = posterior_history_length_

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}

/**
 * @brief Destructor for posterior handler class
 */
PosteriorHandler::~PosteriorHandler() {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  /*
   * Hints:
   * - Every data structure allocated in the constructor above has to be cleaned up properly
   * - This can for example be achieved using free()
   */

  // Free the memory of the posterior history.
  for (uint32_t i = 0; i < posterior_category_count_; ++i)
  {
    delete[] posterior_history_[i];
  }

  // When posterior_history_ is created with new uint8_t*[category_count], 
  // it dynamically allocates a block of memory to store pointers. 
  // Each of these pointers then points to its own dynamically allocated uint8_t array. 
  // Hence, there are two levels of dynamic allocation. We deleted the first level above 
  // (an array of pointers), and now we need to delete the memory allocation that we use 
  // to store that array of pointers
  delete[] posterior_history_;

  // Free the memory of the moving average.
  delete[] moving_average_;

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}

/**
 * @brief Implementation of posterior hanlding algorithm.
 *
 * @param new_posteriors The raw model outputs with unsigned 8-bit values.
 * @param time_ms Timestamp for posterior handling (ms).
 * @param top_category_index The index of the detected category/label returned by pointer.
 * @param trigger Flag which should be raised to true if a new detection is available.
 *
 * @return ESP_OK if no error occured.
 */
esp_err_t PosteriorHandler::Handle(uint8_t* new_posteriors, uint32_t time_ms,
                                   size_t* top_category_index, bool* trigger) {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  /*
   * Hints:
   * - The goal is to implement a posterior handling algorithm descibed in Figure 2.1
   *   and section 2.2.1 of the Lab 2 manual.
   * - By using a moving average over the model outputs, we want to reduce the number
   *   of incorrect classifications i.e. caused by random spikes.
   * - If the calculated moving average for a class exceeds the trigger threshold a
   *   detection should be triggered (unless the deactivation period for a past detection
   *   is still active)
   * - The trigger should be raised for all classes (including silence and unknown) since
   *   the KeywordCallback method in backend.cc is reponsible for deciding which labels should
   *   be ignored.
   * - The supression time (in ms) defines the duration in which registered labels shall
   *   not trigger a new detection. (However their moving average should continue to be updated)
   * - Only if a detection was classified (outside of the deactivation period) the trigger argument
   *   should be set to true by the algorithm
   * - If trigger is high, the detected category index has to updated as well using the argument.
   * - You are allowed (and required) to introduce class variables inside include/posterior.h which
   * may than be (de-)initialized in the constructor/destuctor above.
   */

  // First, update the moving averages for each category
  for (uint8_t i = 0; i < posterior_category_count_; ++i) 
  {
    // Calculate the new moving average value for this category
    uint32_t sum = 0;
    for (uint8_t j = 0; j < posterior_history_length_; ++j)
    {
      sum += posterior_history_[i][j];
      // Shift the history to make room for the new value
      if (j < posterior_history_length_ - 1) 
      {
        posterior_history_[i][j] = posterior_history_[i][j + 1];
      }
    }
      
    // Add the new posterior to the history
    posterior_history_[i][posterior_history_length_ - 1] = new_posteriors[i];

    // Update the moving average with the new sum
    //moving_average_[i] = sum / posterior_history_length_; We dont divide by posterior_history_length_ since we are comparing against
    //trigger_threshold_single * history_length (the cumulative trigger value and not the single trigger value)
    moving_average_[i] = sum;
  }

  // Next, determine the top-scoring class using the updated moving averages
  uint32_t max_index = 0;
  uint32_t max_value = moving_average_[0];
  for (uint8_t i = 1; i < posterior_category_count_; ++i) 
  {
    if (moving_average_[i] > max_value) 
    {
      max_value = moving_average_[i];
      max_index = i;
    }
  }
  //printf("%" PRIu32 "\n", max_value);


  // Trigger a detection if the max value exceeds the threshold and it's not suppressed
  if ((max_value > posterior_trigger_threshold_) && ((time_ms - last_detection_time_) > posterior_suppression_ms_))
  {
    *trigger = true;
    last_detection_time_ = time_ms; // Update the last detection time
  }
  else
  {
    *trigger = false;
  }

  // Set the top category index to the class with the highest moving average
  *top_category_index = max_index;

  return ESP_OK;


  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}
