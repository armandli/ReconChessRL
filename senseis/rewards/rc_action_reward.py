def rc_action_reward1(self_capture_count, oppo_capture_count, terminal, win=False):
  r = 0. + self_capture_count - oppo_capture_count
  if terminal:
    r += 40. if win else -40
  return r
