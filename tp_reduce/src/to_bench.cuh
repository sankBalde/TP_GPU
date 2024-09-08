#pragma once

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total);

void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total);