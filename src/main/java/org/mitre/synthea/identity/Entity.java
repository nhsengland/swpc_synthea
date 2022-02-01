package org.mitre.synthea.identity;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;

public class Entity {
  private List<Seed> seeds;
  private LocalDate dateOfBirth;
  private String gender;
  private String individualId;

  public Entity() {
    this.seeds = new ArrayList<>();
  }

  public List<Seed> getSeeds() {
    return seeds;
  }

  public void setSeeds(List<Seed> seeds) {
    this.seeds = seeds;
  }

  public Seed seedAt(LocalDate date) {
    return seeds.stream().filter(s -> s.getPeriod().contains(date)).findFirst().orElse(null);
  }

  /**
   * Find the seed at a particular time.
   * @param timestamp the time to find a seed
   * @return The seed that covers the time. If before the first seed, will still return the first
   *     seed
   */
  public Seed seedAt(long timestamp) {
    if (timestamp == Long.MIN_VALUE) {
      return seeds.get(0);
    }
    LocalDate date = LocalDateTime.from(Instant.ofEpochMilli(timestamp)
        .atZone(ZoneId.systemDefault())).toLocalDate();
    return seedAt(date);
  }

  public boolean isBeforeOrDuringFirstSeed(long timestamp) {
    Seed firstSeed = seeds.get(0);
    return firstSeed.getPeriod().isBefore(timestamp) || firstSeed.getPeriod().contains(timestamp);
  }

  public LocalDate getDateOfBirth() {
    return dateOfBirth;
  }

  public void setDateOfBirth(LocalDate dateOfBirth) {
    this.dateOfBirth = dateOfBirth;
  }

  public String getGender() {
    return gender;
  }

  public void setGender(String gender) {
    this.gender = gender;
  }

  public String getIndividualId() {
    return individualId;
  }

  public void setIndividualId(String individualId) {
    this.individualId = individualId;
  }
}